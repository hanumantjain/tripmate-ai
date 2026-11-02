from tools.tavliy_tool import tavily_search
from tools.flight_tool import search_flights
from backend import run_travel_agent
# res = tavily_search("Best hotels in Pune")
# print(res)

# res = search_flights("Plan a 7 days Japan trip from India")
# print(res)

user_input = input("Enter your travel query: ")
response = run_travel_agent(
    user_input= user_input,
    thread_id= "test_user",
)

print("\nFinal Itinerary:\n")
print(response["answer"])