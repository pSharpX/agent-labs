from pydantic import BaseModel

class AnswerWithJustification(BaseModel):
    """An answer to the user's question along with justification for the answer."""
    answer: str
    '''The answer to the user's question'''
    justification: str
    '''Justification for the answer'''