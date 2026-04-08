from .Planner.agent import Planner

# Expose the planner directly as the app's root agent so `adk web` chats with it
# without an intermediate router.
root_agent = Planner
