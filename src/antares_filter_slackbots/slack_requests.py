# All Slack calls that require the authorization tokens.

import requests as req
from antares_filter_slackbots.auth import toku, signing_secret, user_toku
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

def setup_user_client():
    """Set up slack web client.
    """
    return WebClient(token=user_toku)

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


def send_slack_message(client, channel, attachments, fallback_text=""):
    if len(attachments) > 50:
        attachments_send = attachments[:50]
        remainder = attachments[50:]

        send_slack_message(client, channel, attachments_send)
        send_slack_message(client, channel, remainder)
    
    else:
        # Sending the message to Slack
        for _ in range(10):
            response = client.chat_postMessage(
                channel=channel,
                text=fallback_text,
                blocks=attachments
            )

            if response['ok']:
                print("Message posted successfully.")
                break
            else:
                print(f"Error posting message: {response['error']}. Trying again...")