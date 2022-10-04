from pydantic import BaseModel
    
class TagCreate(BaseModel):
    user_tags: str
    
class TagRequest(BaseModel):
    id: int
    user_tags: str
    
    class Config:
        orm_mode = True   