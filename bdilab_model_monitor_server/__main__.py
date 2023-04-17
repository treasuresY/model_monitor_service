import argparse
from bdilab_model_monitor_server.server import CEServer
from bdilab_model_monitor_server.protocols import Protocol
from bdilab_model_monitor_server.constants import DEFAULT_HTTP_PORT, DEFAULT_MODEL_NAME
from enum import Enum
from bdilab_model_monitor_server.cm_model import CustomMetricsModel


class BdilabDetectMethod(Enum):
    classification_metric = "ClassificationMetric"
    regression_metric = "RegressionMetric"

    def __str__(self):
        return self.value


parser = argparse.ArgumentParser(description="Parse command line parameters")
parser.add_argument(
    "--http_port",
    default=DEFAULT_HTTP_PORT,
    type=int,
    help="The HTTP Port listened to by the model server.",
)
parser.add_argument(
    "--protocol",
    type=Protocol,
    choices=list(Protocol),
    default="common.http",
    help="The protocol served by the model server",
)
parser.add_argument(
    "--reply_url", type=str, default="", help="URL to send reply cloudevent"
)
parser.add_argument(
    "--event_source", type=str, default="1", help="URI of the event source"
)
parser.add_argument(
    "--event_type",
    type=str,
    default="1",
    help="e.g. io.org.kubeflow.serving.inference.outlier",
)
parser.add_argument(
    "--model_name",
    default=DEFAULT_MODEL_NAME,
    help="The name that the model is served under.",
)

args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = CustomMetricsModel(args.model_name)
    CEServer(
        args.protocol,
        args.event_type,
        args.event_source,
        http_port=args.http_port,
        reply_url=args.reply_url,
    ).start(model)
