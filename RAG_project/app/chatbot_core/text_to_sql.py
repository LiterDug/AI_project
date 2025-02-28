import warnings
warnings.filterwarnings("ignore")
from langchain_community.utilities import SQLDatabase
import  sys
sys.path.append(".")
from models.models import Model_loader
from langchain.prompts import ChatPromptTemplate


# Kết nối đến database SQLite
db_uri = "sqlite:///database/farm_inventory.db"
db = SQLDatabase.from_uri(db_uri)
schema_info = db.get_table_info()

sql_prompt = ChatPromptTemplate.from_template(
    "You are an SQL expert. You perform the following tasks: \n"
    "Given the following database schema:\n{schema_info}\n\n"
    "Generate a SQL query to answer the following question:\n{question}\n\n"
    "If the question cannot be answered using this schema, respond with 'No relevant data'.\n"
    "The query should be a valid SQL statement starting with SELECT.\n"
    "Make sure to properly quote column names with double quotes, especially if they contain spaces or special characters.\n"
    "Return the SQL query and the table name in the format: SQL_QUERY: <your_sql_query_here> ||| TABLE_NAME: <table_name_here>\n"
    "Example: SQL_QUERY: SELECT * FROM \"table_name\" WHERE column_name = 'value' ||| TABLE_NAME: table_name\n"
    "Response:"
)

answer_prompt = ChatPromptTemplate.from_template(
    """
    Bạn là một trợ lý AI hữu ích, trả lời kết quả truy vấn cơ sở dữ liệu.
    Với truy vấn SQL:
    {query}

    Và kết quả truy vấn sau:
    {results}

    Hãy trả lời cho người dùng chi tiết và dễ hiểu bằng ngôn ngữ tự nhiên về kết quả.
    Nếu tập kết quả rỗng, hãy trả lời 'Không tìm thấy dữ liệu liên quan.'.
    Phản hồi:
    """
)
llm_model = Model_loader.load_gemini(1)

def generate_sql(question: str) -> str:
    """Tạo truy vấn SQL từ câu hỏi bằng cách sử dụng mô hình LLM."""
    prompt_messages = [
        {"role": "system", "content": "You are an SQL expert."},
        {"role": "user", "content": sql_prompt.format(schema_info=schema_info, question=question)}
    ]
    response = llm_model.invoke(prompt_messages)
    sql_query = response.content.split(" ||| ")[0].replace("SQL_QUERY: ", "")
    return sql_query


def execute_sql(query: str):
    """Thực thi truy vấn SQL và trả về kết quả."""
    try:
        result = db.run(query)
        return result
    except Exception as e:
        return str(e)


def generate_answer(query: str, results: str):
    """Tạo câu trả lời từ kết quả truy vấn SQL."""
    prompt = answer_prompt.format(query=query, results=results)
    response = llm_model.invoke(prompt)
    return response.content


if __name__ == "__main__":
    query_question = "Còn bao nhiêu phân bón NPK trong kho?"
    query_response = generate_sql(query_question)
    print(query_response)
    # result = execute_sql(query_response)
    # print(result)
    # bot_answer = generate_answer(query_question, result)
    # print(bot_answer)

    

