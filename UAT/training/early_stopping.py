# a basic early stopping test routine based on test set loss
def create_early_stopping(step_start, counter_val, metric_name="loss", tol=1e-8):
    def early_stopping(step, params_new, params_store, test_dict, metrics_store):
        if step <= step_start:
            metrics_store = {}
            for key, value in test_dict.items():
                metrics_store[key] = value
            metrics_store["counter"] = 0
            return False, params_new, metrics_store
        else:
            if (test_dict[metric_name] + tol) <= metrics_store[metric_name]:
                for key, value in test_dict.items():
                    metrics_store[key] = value
                metrics_store["counter"] = 0 
                return False, params_new, metrics_store
            else:
                metrics_store["counter"] += 1
        
        if metrics_store["counter"] > counter_val:
            return True, params_store, metrics_store
        else:
            return False, params_new, metrics_store
    return early_stopping 