# All Slack calls that require the authorization tokens.

import requests as req
from auth import toku, signing_secret
from slack_bolt import App
from slack_sdk import WebClient

def setup_app():
    """Set up slack bolt app.
    """
    return App(
        token=toku, signing_secret=signing_secret,
    )

def setup_client():
    """Set up slack web client.
    """
    return WebClient(token=toku)

def get_conversation_history(channel):
    p1 = req.get(
        'https://slack.com/api/conversations.history',
        params={'channel': channel, 'parse':'none'},
        headers={'Authorization': f'Bearer {toku}'}
    )

    p1.raise_for_status()

    # Additional check for Slack-specific errors in the response body
    response_data = p1.json()
    if not response_data.get('ok'):
        raise ValueError(f"Failed to check posting history: {response_data.get('error')}")
    
    return response_data


def send_slack_message(client, channel, attachments):
    # Sending the message to Slack
    response = client.chat_postMessage(
        channel=channel,
        text="Fallback text",  # This is plain text for clients that don’t support blocks
        blocks=attachments
    )

    if response['ok']:
        print("Message posted successfully.")
    else:
        print(f"Error posting message: {response['error']}")