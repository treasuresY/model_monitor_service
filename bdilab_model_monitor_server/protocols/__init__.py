from enum import Enum


class Protocol(Enum):
    common_http = "common.http"

    def __str__(self):
        return self.value
