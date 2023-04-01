from typing import Union, List, Dict
import tornado
from http import HTTPStatus
from bdilab_model_monitor_server.protocols.request_handler import RequestHandler


class CommonRequestHandler(RequestHandler):

    def __init__(self, request: Dict):
        super().__init__(request)

    def validate(self):
        if "predictions" not in self.request:
            raise Exception("Expected key 'predictions' in request body")
        if "truth" not in self.request:
            raise Exception("Expected key 'truth' in request body")
        if "task_type" not in self.request:
            raise Exception("Expected key 'task_type' in request body")


    def extract_request(self) -> Union[List, Dict]:
        return self.request["predictions"], self.request["truth"], \
               self.request["task_type"]
    # def validate(self):
    #     if "predictions" not in self.request:
    #         raise Exception("Expected key 'predictions' in request body")
    #
    #     if "truth" not in self.request:
    #         raise Exception("Expected key 'truth' in request body")
    #     if "task_type" not in self.request:
    #         raise Exception("Expected key 'task_type' in request body")
    #     if "metrics_type" not in self.request:
    #         raise Exception("Expected key 'metrics_type' in request body")
    #
    # def extract_request(self) -> Union[List, Dict]:
    #     return self.request["predictions"], self.request["truth"], \
    #         self.request["task_type"], self.request["metrics_type"]

