import json
from typing import Union, List, Dict, Optional, Any

from bdilab_model_monitor_server.base import ModelResponse
from bdilab_model_monitor_server.base.model import CEModel
from sklearn import metrics
import numpy as np
from bdilab_model_monitor_server.numpy_encoder import NumpyEncoder
from enum import Enum


class TaskType(Enum):
    classification = 1  # 适用于二分类、多标签二分类、多类别二分类任务
    regression = 2

    def __str__(self):
        return self.value


class Average(Enum):
    binary = 'binary'
    micro = 'micro'
    macro = 'macro'
    weighted = 'weighted'
    samples = 'samples'

    def __str__(self):
        return self.value


class Multioutput(Enum):
    uniform_average = 'uniform_average'
    raw_values = 'raw_values'
    variance_weighted = 'variance_weighted'

    def __str__(self):
        return self.value

def _append_model_monitor_metrcs(metrics, model_monitor, name):
    metric_found = model_monitor.get(name)
    # Assumes metric_found is always float/int or list/np.array when not none
    if metric_found is not None:
        if not isinstance(metric_found, (list, np.ndarray)):
            metric_found = [metric_found]

        for i, instance in enumerate(metric_found):
            metrics.append(
                {
                    "key": f"bdilab_metric_{name}",
                    "value": instance,
                    "type": "GAUGE",
                    "tags": {"index": str(i)},
                }
            )


class CustomMetricsModel(CEModel):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False

    def load(self):
        self.ready = True

    def process_event(self, inputs: Union[List, Dict], headers: Dict) -> Optional[ModelResponse]:
        task_type = headers["task_type"]
        y_true = inputs["y_true"]
        y_pred = inputs["y_pred"]
        output = {}
        output["data"] = {}
        if task_type == TaskType.classification.value:
            output["data"]["multilabel_confusion_matrix"] = self.get_multilabel_confusion_matrix(self, y_true=y_true, y_pred=y_pred)
            output["data"]["accuracy_score"] = self.get_accuracy_score(self, y_true=y_true, y_pred=y_pred)

            output["data"]["precision_score_each"] = self.get_precision_score(self, y_true=y_true, y_pred=y_pred, average=None)
            output["data"]["precision_score_micro"] = self.get_precision_score(self, y_true=y_true, y_pred=y_pred, average=Average.micro.value)
            output["data"]["precision_score_macro"] = self.get_precision_score(self, y_true=y_true, y_pred=y_pred, average=Average.macro.value)
            output["data"]["precision_score_weighted"] = self.get_precision_score(self, y_true=y_true, y_pred=y_pred, average=Average.weighted.value)

            output["data"]["recall_score_each"] = self.get_recall_score(self, y_true=y_true, y_pred=y_pred, average=None)
            output["data"]["recall_score_micro"] = self.get_recall_score(self, y_true=y_true, y_pred=y_pred, average=Average.micro.value)
            output["data"]["recall_score_macro"] = self.get_recall_score(self, y_true=y_true, y_pred=y_pred, average=Average.macro.value)
            output["data"]["recall_score_weighted"] = self.get_recall_score(self, y_true=y_true, y_pred=y_pred, average=Average.weighted.value)

            output["data"]["f1_score_each"] = self.get_f1_score(self, y_true=y_true, y_pred=y_pred, average=None)
            output["data"]["f1_score_micro"] = self.get_f1_score(self, y_true=y_true, y_pred=y_pred, average=Average.micro.value)
            output["data"]["f1_score_macro"] = self.get_f1_score(self, y_true=y_true, y_pred=y_pred, average=Average.macro.value)
            output["data"]["f1_score_weighted"] = self.get_f1_score(self, y_true=y_true, y_pred=y_pred, average=Average.weighted.value)

            output["data"]["roc_auc_score_each"] = self.get_roc_auc_score(self, y_true=y_true, y_pred=y_pred, average=None)
            output["data"]["roc_auc_score_micro"] = self.get_roc_auc_score(self, y_true=y_true, y_pred=y_pred, average=Average.micro.value)
            output["data"]["roc_auc_score_macro"] = self.get_roc_auc_score(self, y_true=y_true, y_pred=y_pred, average=Average.macro.value)
            output["data"]["roc_auc_score_weighted"] = self.get_roc_auc_score(self, y_true=y_true, y_pred=y_pred, average=Average.weighted.value)

            output["data"]["log_loss"] = self.get_log_loss(self, y_true=y_true, y_pred=y_pred)

            output["data"]["balanced_accuracy_score"] = self.get_balanced_accuracy_score(self, y_true=y_true, y_pred=y_pred)
            output["data"]["confusion_matrix"] = self.get_confusion_matrix(self, y_true=y_true, y_pred=y_pred)
            output["data"]["matthews_corrcoef"] = self.get_matthews_corrcoef(self, y_true=y_true, y_pred=y_pred)
        elif task_type == TaskType.regression.value:
            output["data"]["explained_variance_score_uniform_average"] = self.get_explained_variance_score(self, y_true=y_true, y_pred=y_pred, multioutput=Multioutput.uniform_average.value)
            output["data"]["explained_variance_score_raw_values"] = self.get_explained_variance_score(self, y_true=y_true, y_pred=y_pred, multioutput=Multioutput.raw_values.value)

            output["data"]["mean_absolute_error_uniform_average"] = self.get_mean_absolute_error(self, y_true=y_true, y_pred=y_pred, multioutput=Multioutput.uniform_average.value)
            output["data"]["mean_absolute_error_raw_values"] = self.get_mean_absolute_error(self, y_true=y_true, y_pred=y_pred, multioutput=Multioutput.raw_values.value)

            output["data"]["mean_squared_uniform_average"] = self.get_mean_squared_error(self, y_true=y_true, y_pred=y_pred, multioutput=Multioutput.uniform_average.value, squared=True)
            output["data"]["mean_squared_error_raw_values"] = self.get_mean_squared_error(self, y_true=y_true, y_pred=y_pred, multioutput=Multioutput.raw_values.value, squared=True)
            output["data"]["root_mean_squared_error_uniform_average"] = self.get_mean_squared_error(self, y_true=y_true, y_pred=y_pred, multioutput=Multioutput.uniform_average.value, squared=False)
            output["data"]["root_mean_squared_error_raw_values"] = self.get_mean_squared_error(self, y_true=y_true, y_pred=y_pred, multioutput=Multioutput.raw_values.value, squared=False)

            output["data"]["mean_squared_log_error_uniform_average"] = self.get_mean_squared_log_error(self, y_true=y_true, y_pred=y_pred, multioutput=Multioutput.uniform_average.value, squared=True)
            output["data"]["mean_squared_log_error_raw_values"] = self.get_mean_squared_log_error(self, y_true=y_true, y_pred=y_pred, multioutput=Multioutput.raw_values.value, squared=True)
            output["data"]["root_mean_squared_log_error_uniform_average"] = self.get_mean_squared_log_error(self, y_true=y_true, y_pred=y_pred, multioutput=Multioutput.uniform_average.value, squared=False)
            output["data"]["root_mean_squared_log_error_raw_values"] = self.get_mean_squared_log_error(self, y_true=y_true, y_pred=y_pred, multioutput=Multioutput.raw_values.value, squared=False)

            output["data"]["median_absolute_error_uniform_average"] = self.get_median_absolute_error(self, y_true=y_true, y_pred=y_pred, multioutput=Multioutput.uniform_average.value)
            output["data"]["median_absolute_error_raw_values"] = self.get_median_absolute_error(self, y_true=y_true, y_pred=y_pred, multioutput=Multioutput.raw_values.value)

            output["data"]["mean_absolute_percentage_error"] = self.get_mean_absolute_percentage_error(self, y_true=y_true, y_pred=y_pred, multioutput=Multioutput.uniform_average.value)
            output["data"]["mean_absolute_percentage_error_raw"] = self.get_mean_absolute_percentage_error(self, y_true=y_true, y_pred=y_pred, multioutput=Multioutput.raw_values.value)

            output["data"]["r2_score_uniform_average"] = self.get_r2_score(self, y_true=y_true, y_pred=y_pred, multioutput=Multioutput.uniform_average.value)
            output["data"]["r2_score_raw_values"] = self.get_r2_score(self, y_true=y_true, y_pred=y_pred, multioutput=Multioutput.raw_values.value)
        else:
            raise Exception("暂不支持此种类型的模型监控指标")

        # 删除值为 None 的键值对
        cleaned_output = {}
        cleaned_output["data"] = {k: v for k, v in output["data"].items() if v is not None}

        # 度量指标向prometheus公开
        exclude_metric = list()
        exclude_metric.append("multilabel_confusion_matrix")
        exclude_metric.append("confusion_matrix")
        exclude_metric.append("precision_score_each")
        exclude_metric.append("recall_score_each")
        exclude_metric.append("f1_score_each")
        exclude_metric.append("roc_auc_score_each")
        metrics: List[Dict] = []
        for k in cleaned_output["data"].keys():
            if k not in exclude_metric:    # 此类指标结果无法转换为有效的Prometheus指标类型格式
                _append_model_monitor_metrcs(metrics, cleaned_output["data"], k)

        res_data = json.loads(json.dumps(cleaned_output, cls=NumpyEncoder))
        response = ModelResponse(data=res_data, metrics=metrics)
        return response


    @staticmethod
    def get_precision_recall_curve(self, y_true: Union[List], y_pred: Union[List],
                                   pos_label: Union[int, str] = None, sample_weight: Union[List] = None):
        try:
            precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred)
        except Exception as e:
            print("Oops! An error occurred:", e)
            return None
        return precision, recall, thresholds

    @staticmethod
    def get_roc_curve(self, y_true: Union[List], y_pred: Union[List]):
        try:
            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
        except Exception as e:
            print("Oops! An error occurred:", e)
            return None
        return fpr, tpr, thresholds

    @staticmethod
    def get_accuracy_score(self, y_true: Union[List], y_pred: Union[List]):
        try:
            return metrics.accuracy_score(y_true, y_pred, normalize=True)
        except Exception as e:
            return None

    @staticmethod
    def get_balanced_accuracy_score(self, y_true: Union[List], y_pred: Union[List]):
        try:
            return metrics.balanced_accuracy_score(y_true, y_pred)
        except Exception as e:
            return None

    @staticmethod
    def get_precision_score(self, y_true: Union[List], y_pred: Union[List], average: Any):
        try:
            return metrics.precision_score(y_true, y_pred, average=average, zero_division=0)
        except Exception as e:
            return None

    @staticmethod
    def get_average_precision_score(self, y_true: Union[List], y_pred: Union[List]):
        try:
            return metrics.average_precision_score(y_true, y_pred)
        except Exception as e:
            return None

    @staticmethod
    def get_recall_score(self, y_true: Union[List], y_pred: Union[List], average: Any):
        try:
            return metrics.recall_score(y_true, y_pred, average=average, zero_division=0)
        except Exception as e:
            return None

    @staticmethod
    def get_average_recall_score(self, y_true: Union[List], y_pred: Union[List]):
        try:
            return metrics.average_recall_score(y_true, y_pred, average="macro", zero_division=0)
        except Exception as e:
            return None

    @staticmethod
    def get_f1_score(self, y_true: Union[List], y_pred: Union[List], average:Any):
        try:
            return metrics.f1_score(y_true, y_pred, average=average, zero_division=0)
        except Exception as e:
            return None

    @staticmethod
    def get_fbeta_score(self, y_true: Union[List], y_pred: Union[List]):
        try:
            return metrics.fbeta_score(y_true, y_pred, average=None, zero_division=0)
        except Exception as e:
            return None

    @staticmethod
    def get_average_f1_score(self, y_true: Union[List], y_pred: Union[List]):
        try:
            return metrics.average_f1_score(y_true, y_pred, average='macro', zero_division=0)
        except Exception as e:
            return None

    @staticmethod
    def get_log_loss(self, y_true: Union[List], y_pred: Union[List]):
        try:
            return metrics.log_loss(y_true, y_pred)
        except Exception as e:
            return None

    @staticmethod
    def get_confusion_matrix(self, y_true: Union[List], y_pred: Union[List]):
        try:
            return metrics.confusion_matrix(y_true, y_pred)
        except Exception as e:
            return None

    @staticmethod
    def get_multilabel_confusion_matrix(self, y_true: Union[List], y_pred: Union[List]):
        try:
            return metrics.multilabel_confusion_matrix(y_true, y_pred)
        except Exception as e:
            return None

    @staticmethod
    def get_matthews_corrcoef(self, y_true: Union[List], y_pred: Union[List]):
        try:
            return metrics.matthews_corrcoef(y_true, y_pred)
        except Exception as e:
            return None

    @staticmethod
    def get_roc_auc_score(self, y_true: Union[List], y_pred: Union[List], average: Any):
        try:
            return metrics.roc_auc_score(y_true, y_pred, average=average)
        except Exception as e:
            return None

    # 分水岭

    @staticmethod
    def get_top_k_accuracy_score(self, y_true: Union[List], y_pred: Union[List]):
        try:
            top_k_accuracy_score = metrics.top_k_accuracy_score(y_true, y_pred)
        except Exception as e:
            print("Oops! An error occurred:", e)
            return None
        return top_k_accuracy_score

    @staticmethod
    def get_brier_score_loss(self, y_true: Union[List], y_pred: Union[List]):
        try:
            brier_score_loss = metrics.brier_score_loss(y_true, y_pred)
        except Exception as e:
            print("Oops! An error occurred:", e)
            return None
        return brier_score_loss

    @staticmethod
    def get_precision_recall_fscore_support(self, y_true: Union[List], y_pred: Union[List]):
        try:
            precision_recall_fscore_support = metrics.precision_recall_fscore_support(y_true, y_pred)
        except Exception as e:
            print("Oops! An error occurred:", e)
            return None
        return precision_recall_fscore_support

    # 回归任务指标
    @staticmethod
    def get_mean_squared_error(self, y_true: Union[List], y_pred: Union[List], multioutput, squared):
        try:
            return metrics.mean_squared_error(y_true, y_pred, multioutput=multioutput, squared=squared)
        except Exception as e:
            return None

    @staticmethod
    def get_mean_absolute_error(self, y_true: Union[List], y_pred: Union[List], multioutput: Any):
        try:
            return metrics.mean_absolute_error(y_true, y_pred, multioutput=multioutput)
        except Exception as e:
            return None


    @staticmethod
    def get_mean_absolute_percentage_error(self, y_true: Union[List], y_pred: Union[List], multioutput):
        try:
            return metrics.mean_absolute_percentage_error(y_true, y_pred, multioutput=multioutput)
        except Exception as e:
            return None

    @staticmethod
    def get_explained_variance_score(self, y_true: Union[List], y_pred: Union[List], multioutput: Any):
        try:
            return metrics.explained_variance_score(y_true, y_pred, multioutput=multioutput)
        except Exception as e:
            return None


    @staticmethod
    def get_r2_score(self, y_true: Union[List], y_pred: Union[List], multioutput):
        try:
            return metrics.r2_score(y_true, y_pred, multioutput=multioutput)
        except Exception as e:
            return None

    # 分水岭

    @staticmethod
    def get_mean_squared_log_error(self, y_true: Union[List], y_pred: Union[List], multioutput, squared):
        try:
            return metrics.mean_squared_log_error(y_true, y_pred, multioutput=multioutput, squared=squared)
        except Exception as e:
            return None

    @staticmethod
    def get_mean_poisson_deviance(self, y_true: Union[List], y_pred: Union[List]):
        try:
            mean_poisson_deviance = metrics.mean_poisson_deviance(y_true, y_pred)
        except Exception as e:
            print("Oops! An error occurred:", e)
            return None
        return mean_poisson_deviance

    @staticmethod
    def get_mean_gamma_deviance(self, y_true: Union[List], y_pred: Union[List]):
        try:
            mean_gamma_deviance = metrics.mean_gamma_deviance(y_true, y_pred)
        except Exception as e:
            print("Oops! An error occurred:", e)
            return None
        return mean_gamma_deviance

    @staticmethod
    def get_mean_tweedie_deviance(self, y_true: Union[List], y_pred: Union[List]):
        try:
            mean_tweedie_deviance = metrics.mean_tweedie_deviance(y_true, y_pred)
        except Exception as e:
            print("Oops! An error occurred:", e)
            return None
        return mean_tweedie_deviance

    @staticmethod
    def get_max_error(self, y_true: Union[List], y_pred: Union[List]):
        try:
            max_error = metrics.max_error(y_true, y_pred)
        except Exception as e:
            print("Oops! An error occurred:", e)
            return None
        return max_error

    @staticmethod
    def get_median_absolute_error(self, y_true: Union[List], y_pred: Union[List], multioutput):
        try:
            return metrics.median_absolute_error(y_true, y_pred, multioutput=multioutput)
        except Exception as e:
            return None

    @staticmethod
    def get_d2_absolute_error_score(self, y_true: Union[List], y_pred: Union[List]):
        try:
            d2_absolute_error_score = metrics.d2_absolute_error_score(y_true, y_pred)
        except Exception as e:
            print("Oops! An error occurred:", e)
            return None
        return d2_absolute_error_score

    @staticmethod
    def get_d2_pinball_score(self, y_true: Union[List], y_pred: Union[List]):
        try:
            d2_pinball_score = metrics.d2_pinball_score(y_true, y_pred)
        except Exception as e:
            print("Oops! An error occurred:", e)
            return None
        return d2_pinball_score

    @staticmethod
    def get_d2_tweedie_score(self, y_true: Union[List], y_pred: Union[List]):
        try:
            d2_tweedie_score = metrics.d2_tweedie_score(y_true, y_pred)
        except Exception as e:
            print("Oops! An error occurred:", e)
            return None
        return d2_tweedie_score

    @staticmethod
    def get_mean_pinball_loss(self, y_true: Union[List], y_pred: Union[List]):
        try:
            mean_pinball_loss = metrics.mean_pinball_loss(y_true, y_pred)
        except Exception as e:
            print("Oops! An error occurred:", e)
            return None
        return mean_pinball_loss








