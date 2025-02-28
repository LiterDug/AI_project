import warnings
warnings.filterwarnings("ignore")
from utils import llm_model
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field 
from typing import Literal


class RouteQuery(BaseModel):
    """Định tuyến câu hỏi người dùng đến các tool phù hợp nhất"""
    tool: Literal["get_retrieval", "get_weather", "get_sql", "google_search", "chitchat"] = Field(
        ...,
        description="Tên của tool cần sử dụng để xử lý câu hỏi của người dùng",
    )


llm = llm_model
structure_llm = llm.with_structured_output(RouteQuery)

system = """
Bạn là một hệ thống có thể định tuyến câu hỏi của người dùng đến các tool phù hợp nhất. Hãy cẩn trọng và định tuyến chính xác nhất.
Ví dụ: 
- Các câu trần thuật, chuyện phiếm sẽ được định tuyến đến tool `chitchat`
- Các câu hỏi về thời tiết sẽ được định tuyến đến tool `get_weather`
- Các câu hỏi về dữ liệu của vật tư nông nghiệp trong kho, nông trại sẽ được định tuyến đến tool `get_sql`
- Các câu hỏi về vật thể, thông tin, sự kiện, vấn đề không liên quan đến nông nghiệp hoặc không liên quan đến các công cụ khác sẽ được định tuyến đến tool `google_search`
- Các câu hỏi về nông nghiệp chẳng hạn như sâu bệnh, cây cối, cách điều trị, cách chăm sóc sẽ được định tuyến đến tool `get_retrieval`
"""


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}")
    ]
)

router = prompt | structure_llm


if __name__ == "__main__":
    question = "thời tiết hôm nay thế nào"
    tool = router.invoke({"question": question}).tool
    print(tool)
