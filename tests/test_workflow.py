"""Offline behavioural checks. Fake API responses; no key or network needed."""
import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import fitz
import numpy as np
import pandas as pd

from termsheet_analyser.config import Config
from termsheet_analyser.step06_parameters import PricingParameters, PricingParameterExtractor, PricingMethodologyAnalyzer
from termsheet_analyser.step07_pricing import UniversalMonteCarloPricer
from termsheet_analyser.workflow import run_workflow

RAW_TABLES = """KEY_TERMS_TABLE
| Term | Value |
| --- | --- |
| Issuer | Synthetic Issuer |
| Coupon Rate | 0.5% per month |
DATES_TABLE
| Date Type | Date | Applies To | Trigger / Barrier Level |
| --- | --- | --- | --- |
| Observation Date | 2027-01-15 | Coupon | 70% |
| Observation Date | 2027-01-15 | Autocall | 100% |
"""
STRUCTURE = {"product_classification":{"type":"Synthetic note"}, "payoff_logic":{"mece_scenarios":[]}, "calendar":[]}


def response(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class WorkflowTests(unittest.TestCase):
    def make_pdf(self, directory, text="Issuer: Synthetic Issuer\nQuarterly autocall note"):
        path=Path(directory)/"synthetic.pdf"
        with fitz.open() as doc:
            page=doc.new_page()
            page.insert_text((40,50),text)
            doc.save(path)
        return path

    def test_all_workflow_modules_import(self):
        for name in ["step01_pdf","step02_extraction","step03_structure","step04_scenarios","step05_visualization","step06_parameters","step07_pricing","step08_reporting"]:
            with self.subTest(module=name):
                importlib.import_module("termsheet_analyser."+name)

    def test_pricing_reports_are_saved_as_files(self):
        from termsheet_analyser import step08_reporting as reports
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(reports, "PricingParameterExtractor") as extractor, \
                 patch.object(reports, "UniversalMonteCarloPricer") as pricer, \
                 patch.object(reports, "generate_pricing_html_report", return_value="<html>Research report</html>"), \
                 patch.object(reports, "generate_comprehensive_json", return_value={"status":"synthetic"}), \
                 patch.object(reports, "generate_comprehensive_pdf", return_value=None):
                extractor.return_value.extract_all.return_value = PricingParameters()
                pricer.return_value.price.return_value = {"fair_value":1.0}
                result = reports.run_task3_pricing(None, None, {}, {}, "sample.pdf", output_directory=tmp)
            self.assertIn("Research report", (Path(tmp)/"comprehensive_report_sample.html").read_text())
            self.assertEqual(json.loads((Path(tmp)/"comprehensive_report_sample.json").read_text()), {"status":"synthetic"})
            self.assertEqual(result["results"], {"fair_value":1.0})

    def test_parameters_are_dataclasses_with_independent_mutables(self):
        a,b=PricingParameters(),PricingParameters()
        a.observation_dates.append("2027-01-01")
        a.initial_levels[0]=25
        self.assertEqual(b.observation_dates,[])
        self.assertEqual(b.initial_levels[0],100)
        self.assertEqual(PricingParameters(notional=1000).notional,1000)

    def test_sub_one_percent_conversion(self):
        parser=PricingParameterExtractor()
        self.assertEqual(parser._extract_percentages("0.5% / 1% / 60%"),[.005,.01,.6])
        self.assertAlmostEqual(parser._parse_coupon_rate("0.5% per month"),.06)

    def test_extractor_does_not_reuse_previous_document(self):
        extractor=PricingParameterExtractor()
        dates=pd.DataFrame([{"Date Type":"Observation Date","Date":"2027-01-01"}])
        first=extractor.extract_all(pd.DataFrame(),dates,{}, {})
        second=extractor.extract_all(pd.DataFrame(),pd.DataFrame(),{}, {})
        self.assertEqual(first.observation_dates,["2027-01-01"])
        self.assertEqual(second.observation_dates,[])

    @patch("termsheet_analyser.step06_parameters.OpenAI")
    def test_methodology_uses_same_openai_compatible_interface(self, constructor):
        payload={"product_category":"FIXED_COUPON","pricing_features":{"has_fixed_coupon":True}}
        create=constructor.return_value.chat.completions.create
        create.return_value=response(json.dumps(payload))
        result=PricingMethodologyAnalyzer(Config(API_KEY="offline-test")).analyze(None,None,{}, {})
        self.assertEqual(result,payload)
        self.assertEqual(create.call_args.kwargs["model"],Config().MODEL)

    @patch("termsheet_analyser.step06_parameters.OpenAI")
    def test_failed_methodology_never_defaults_to_autocall(self, constructor):
        constructor.return_value.chat.completions.create.return_value=response("invalid JSON")
        with self.assertRaisesRegex(ValueError,"pricing stopped"):
            PricingMethodologyAnalyzer(Config(API_KEY="offline-test")).analyze(None,None,{}, {})

    def test_unknown_product_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"Unsupported product"):
            UniversalMonteCarloPricer(PricingParameters(product_category="UNKNOWN"))._calculate_payoffs()

    def test_fixed_coupon_reference_case(self):
        params=PricingParameters(product_category="FIXED_COUPON",notional=100,coupon_rate=.06,
                                 maturity_years=2,risk_free_rate=.03,n_simulations=4,n_steps_per_year=12)
        result=UniversalMonteCarloPricer(params).price()
        self.assertAlmostEqual(result["fair_value"],112*np.exp(-.06))

    def test_api_workflow_preserves_date_event_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf=self.make_pdf(tmp)
            client=MagicMock()
            client.chat.completions.create.side_effect=[response(RAW_TABLES),response(json.dumps(STRUCTURE))]
            output=Path(tmp)/"out"
            result=run_workflow(pdf,output,config=Config(API_KEY="offline-test"),client=client,render_charts=False)
            data=json.loads(result.analysis_file.read_text(encoding="utf-8"))
            self.assertEqual(len(data["dated_events"]),2)
            self.assertEqual([r["Applies To"] for r in data["dated_events"]],["Coupon","Autocall"])
            self.assertFalse(data["pricing_enabled"])
            self.assertIsNone(result.pricing)
            self.assertEqual(client.chat.completions.create.call_count,2)
            self.assertTrue((output/"02_dates.csv").exists())

    def test_malformed_extraction_stops_before_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            client=MagicMock()
            client.chat.completions.create.return_value=response("No tables")
            with self.assertRaisesRegex(ValueError,"Term/Value"):
                run_workflow(self.make_pdf(tmp),Path(tmp)/"out",config=Config(API_KEY="offline-test"),client=client,render_charts=False)
            self.assertEqual(client.chat.completions.create.call_count,1)

    def test_long_pdf_is_not_silently_truncated(self):
        with tempfile.TemporaryDirectory() as tmp:
            client=MagicMock()
            with self.assertRaisesRegex(ValueError,"character limit"):
                run_workflow(self.make_pdf(tmp),Path(tmp)/"out",config=Config(API_KEY="offline-test",MAX_CHARS=10),client=client,render_charts=False)
            client.chat.completions.create.assert_not_called()

    def test_missing_key_fails_before_api(self):
        with self.assertRaisesRegex(ValueError,"TERMSHEET_API_KEY"):
            run_workflow("missing.pdf",config=Config(API_KEY=""))

    def test_credential_not_in_config_repr(self):
        self.assertNotIn("offline-test",repr(Config(API_KEY="offline-test")))

    def test_scenario_chart_is_saved_to_requested_directory(self):
        from termsheet_analyser.step05_visualization import EducationalVisualizer
        metadata={
            "dates":{"start":"2027-01-01","end":"2028-01-01"},
            "calendar":[{"date":"2027-04-01","levels":{"Knock-out":100}},
                        {"date":"2028-01-01","levels":{"Knock-out":100}}],
            "barrier_hierarchy":[{"type":"Knock-out","value":100},
                                 {"type":"Knock-in","value":60}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            image=EducationalVisualizer().plot_professional_scenario(
                {"condition":"safe maturity", "chart_type":"safe_maturity", "outcome":"principal"},
                metadata,output_directory=tmp)
            self.assertEqual(Path(image).parent,Path(tmp))
            self.assertGreater(Path(image).stat().st_size,1000)



if __name__=="__main__":
    unittest.main()
