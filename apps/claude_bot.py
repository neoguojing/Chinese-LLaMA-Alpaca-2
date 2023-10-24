import time  
from slack_sdk import WebClient  
from slack_sdk.errors import SlackApiError  
from pydantic import BaseModel  
from typing import Optional  
  
class ClaudeChatReqDto(BaseModel):  
    prompt: str

class ReturnBase(BaseModel):  
    msg: Optional[str] = "ok"  
    msgCode: Optional[str] = "10000"  
    data: Optional[object] = None


def send_message(client, channel, text):  
    try:  
        return client.chat_postMessage(channel=channel, text=text)  
    except SlackApiError as e:  
        print(f"Error sending message: {e}")  
  
  
def fetch_messages(client, channel, last_message_timestamp, bot_user_id):  
    response = client.conversations_history(channel=channel, oldest=last_message_timestamp)  
    return [msg['text'] for msg in response['messages'] if msg['user'] == bot_user_id]  
  
  
def get_new_messages(client, channel, last_message_timestamp, bot_user_id):  
    while True:  
        messages = fetch_messages(client, channel, last_message_timestamp, bot_user_id)  
        if messages and not messages[-1].endswith('Typing…_'):  
            return messages[-1]  
        time.sleep(1) # 这里的时间设置需要 balance 一下，越小越可能被限流，越大返回时间越长
  
  
def find_direct_message_channel(client, user_id):  
    try:  
        response = client.conversations_open(users=user_id)  
        return response['channel']['id']  
    except SlackApiError as e:  
        print(f"Error opening DM channel: {e}")  
  
  
def claude_chat_by_slack_svc(req_dto: ClaudeChatReqDto):  
    slack_user_token = "" 
    bot_user_id = ""
    client = WebClient(token=slack_user_token)  
    dm_channel_id = find_direct_message_channel(client, bot_user_id)  
    if not dm_channel_id:  
        print("Could not find DM channel with the bot.")  
        return  
  
    last_message_timestamp = None  
    prompt = req_dto.prompt  
    response = send_message(client, dm_channel_id, prompt)  
    if response:  
        last_message_timestamp = response['ts']  
    new_message = get_new_messages(client, dm_channel_id, last_message_timestamp, bot_user_id)  

    return ReturnBase(
        data=new_message  
    )  
  
if __name__ == "__main__":  
    while True:
        raw_input_text = input("Input:")
        if len(raw_input_text.strip())==0:
            continue
        resp = claude_chat_by_slack_svc(ClaudeChatReqDto(prompt=raw_input_text))
        print("Response:",resp.data)
        print("\n")
