class GenerateFailedException(Exception):
    def __init__(self, task):
        self.task =task

    
    def __str__(self):
        return f"Failed to finish {self.task}"