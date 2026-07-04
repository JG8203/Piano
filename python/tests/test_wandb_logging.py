import unittest

from piano_fit import wandb_logging


class FakeRun:
    def __init__(self):
        self.config = {}
        self.defined_metrics = []
        self.logged = []

    def define_metric(self, *args, **kwargs):
        self.defined_metrics.append((args, kwargs))

    def log(self, payload, **kwargs):
        self.logged.append((payload, kwargs))


class WandbLoggingTests(unittest.TestCase):
    def test_log_metric_uses_evaluation_field_without_manual_step(self):
        run = FakeRun()

        wandb_logging.define_wandb_metrics(run)
        wandb_logging.log_metric(
            run,
            {
                "type": "evaluation",
                "evaluation": 16,
                "loss": 39.4310799,
                "best_loss": 29.1897774,
                "phase": "pagmo-pso-batch",
            },
        )

        self.assertIn((("evaluation",), {}), run.defined_metrics)
        self.assertIn((("loss",), {"step_metric": "evaluation"}), run.defined_metrics)
        self.assertEqual(
            [
                (
                    {
                        "evaluation": 16,
                        "loss": 39.4310799,
                        "best_loss": 29.1897774,
                        "phase": "pagmo-pso-batch",
                        "event_type": "evaluation",
                    },
                    {},
                )
            ],
            run.logged,
        )


if __name__ == "__main__":
    unittest.main()
