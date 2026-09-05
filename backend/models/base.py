class BaseModel:
    name = "base"
    def predict(self, *args, **kwargs):
        raise NotImplementedError
