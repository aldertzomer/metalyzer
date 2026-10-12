"""Exploratory paired counts and API usage instrumentation, without network."""
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pandas as pd
from benchmark_smoke.evaluate_noisy_fields import paired_statistics
from modules import mistral
from modules.contracts import SourceVocabulary


class PairedComparisonTests(unittest.TestCase):
    def test_discordance_and_unknown_transitions(self):
        rows=pd.DataFrame(dict(unfiltered_correct=[False,True,False,True,False],
                               filtered_correct=[True,False,False,True,False],
                               changed_prediction=[True,True,True,False,False],
                               unfiltered_best_hit=['unknown','cat','dog','cat','dog'],
                               filtered_best_hit=['cat','unknown','sheep','cat','dog']))
        result=paired_statistics(rows)
        for name in ['wrong_to_correct','correct_to_wrong','wrong_to_different_wrong',
                     'unchanged_correct','unchanged_wrong','unknown_to_assigned','assigned_to_unknown']:
            self.assertEqual(result[name],1,name)
        self.assertEqual(result['mcnemar_exact_p'],1)
        self.assertEqual(result['accuracy_difference'],0)
        self.assertLessEqual(result['paired_bootstrap_95pct_low'],0)
        self.assertGreaterEqual(result['paired_bootstrap_95pct_high'],0)


class APIUsageTests(unittest.IsolatedAsyncioTestCase):
    async def test_request_and_token_counts_without_changing_parameters(self):
        response=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"source":"cat"}'))],
                                 usage=SimpleNamespace(prompt_tokens=123,completion_tokens=7,total_tokens=130))
        client=SimpleNamespace(chat=SimpleNamespace(complete_async=AsyncMock(return_value=response)))
        metrics=dict(requests=0,retries=0,completed=0,prompt_tokens=0,completion_tokens=0,total_tokens=0,usage_available=False)
        result=await mistral.classify_one(client,'host: cat',0,SourceVocabulary(('cat',),'accession'),
                                          mistral.MistralConfig(),'same prompt',{},metrics)
        self.assertEqual(result,('cat',''))
        self.assertEqual(metrics,dict(requests=1,retries=0,completed=1,prompt_tokens=123,completion_tokens=7,total_tokens=130,usage_available=True))
        arguments=client.chat.complete_async.call_args.kwargs
        self.assertEqual(arguments['random_seed'],12345)
        self.assertEqual(arguments['temperature'],0)
        self.assertEqual(arguments['retries'],None)


if __name__=='__main__': unittest.main()
