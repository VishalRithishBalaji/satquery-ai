def confidence_from_result(result): return max(0.0,min(1.0,float(result.get('confidence',0.0))))
