from pydantic import BaseModel


class SystemStatus(BaseModel):
    running: bool
