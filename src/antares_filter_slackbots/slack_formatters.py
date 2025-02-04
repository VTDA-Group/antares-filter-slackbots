from astro_ghost.PS1QueryFunctions import geturl
import numpy as np
import pandas as pd
import os
import re

from flask import Flask, request
from werkzeug.middleware.proxy_fix import ProxyFix
from slack_bolt.adapter.flask import SlackRequestHandler

from antares_filter_slackbots.slack_requests import (
    send_slack_message,
    setup_app,
    setup_client
)

def generate_context_block(df, obj_id):
    """Construct context block with previous votes for object.
    """
    votes = {}

    if df is None:
        return {
            "type": "context",
            "elements": [{
                "type": "plain_text",
                "text": "No votes yet."
            }],
            "block_id": f"votes_{obj_id}"
        }
    
    sub_df = df.loc[df.index == obj_id]

    if len(sub_df) == 0:
        return {
            "type": "context",
            "elements": [{
                "type": "plain_text",
                "text": "No votes yet."
            }],
            "block_id": f"votes_{obj_id}"
        }

    for row in sub_df.itertuples(): # users
        vote = row.Response
        votes[row.UserName] = vote

    votes_text = []
    for (name, vote) in votes.items():
        if vote == 'upvote':
            votes_text.append(
                {
                    "type": "plain_text",
                    "text": f"{name} marked this event as relevant."
                }
            )
        elif vote == 'strong_upvote':
            votes_text.append(
                {
                    "type": "plain_text",
                    "text": f"{name} marked this event as high priority."
                }
            )
        elif vote == 'downvote':
            votes_text.append(
                {
                    "type": "plain_text",
                    "text": f"{name} marked this event as not relevant."
                }
            )
        elif vote == 'AGN':
            votes_text.append(
                {
                    "type": "plain_text",
                    "text": f"{name} marked this event as an AGN."
                }
            )
        else:
            continue

    return {
        "type": "context",
        "elements": votes_text,
        "block_id": f"votes_{obj_id}"
    }

class SlackPoster:
    def __init__(self, loci_df, filt_meta, save_prefix):
        '''
        Reformats locus information for Slack posting.

        Parameters
        ----------
        loci_df: pd.DataFrame
            Stores extracted locus features to display with Slack bot
        '''
        self._client = setup_client()
        self._ziggy_url_base = 'https://ziggy.ucolick.org/yse/transient_detail/'
        self._antares_url_base = 'https://antares.noirlab.edu/loci/'
        self._df = loci_df
        self._meta = filt_meta

        # Where to save voting history
        self._votes_fn = os.path.join(save_prefix, "voting_history.csv")
        self.filter_name = save_prefix.split("/")[-1]
        
        if os.path.exists(self._votes_fn):
            self._vote_df = pd.read_csv(self._votes_fn, index_col=0)
        else:
            self._vote_df = None

    def round_sigfigs(self, x, sig=4):
        """Round numerics to sig significant figures.
        """
        if isinstance(x, (int, float, np.number)):  # Check for numeric types
            if (x == 0):  # Handle zero separately to avoid log10 issues
                return x
            elif np.isnan(x):
                return '---'
            else:
                return round(x, -int(np.floor(np.log10(abs(x)))) + (sig - 1))
        elif isinstance(x, str) and x == '':
            return '---'
        elif x is None:
            return '---'
        
        return x  # Return as-is if not numeric

    def unsnake(self, col):
        """Convert column names from snake case
        to normal spacing/capitalization.
        """
        c_split = col.split("_")
        c_capitalized = [word.capitalize() for word in c_split]
        c_reformat = ' '.join(c_capitalized)
        return c_reformat

    def ps1_pic(self, row):
        """Retrive PS1 image url of entry."""
        if row.dec > -30:
            for _ in range(5): # 5 attempts
                try:
                    return geturl(row.ra, row.dec, color=True)
                except:
                    continue
        
        return None
       
    def voting_action(self, suffix):
        return {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": " :yesagn: "
                    },
                    "value": "AGN",
                    "action_id": f"vote$agn${self.filter_name}${suffix}"
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": " :thumbsup: "
                    },
                    "value": "upvote",
                    "action_id": f"vote$upvote${self.filter_name}${suffix}"
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": " :thumbsup::thumbsup: "
                    },
                    "value": "strong_upvote",
                    "action_id": f"vote$strongupvote${self.filter_name}${suffix}"
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": " :thumbsdown: "
                    },
                    "value": "downvote",
                    "action_id": f"vote$downvote${self.filter_name}${suffix}"
                }
            ]
        }
    
    def generate_single_row_attachment(self, row):
        """Generate attachment for single row/locus."""

        title = f':collision: {row.name} :collision:'
        title_link = self._antares_url_base + row.name

        """
        if not row.posted_before:
            color = "#36a64f"
        else: 
            color = "#6D2E9E"
        """

        attachment = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"<{title_link}|*{title}*>"
                }
            },
        ]

        attachment.append(
            {
                "type": "image",
                "image_url": self.ps1_pic(row),
                "alt_text": "ps1_image"
            }
        )

        fields = {
            "type": "section",
            "fields": []
        }

        if row.tns_name != '---':
            tns_url = self._ziggy_url_base + row.tns_name
            fields['fields'] = [
                {"type": "mrkdwn", "text": f"*TNS Name*: <{tns_url}|{row.tns_name}>"},
                {"type": "mrkdwn", "text": f"*TNS Spec. Class*: {row.tns_class}"},
                {"type": "mrkdwn", "text": f"*TNS Redshift*: {row.tns_redshift}"},
            ]

        fields['fields'].append(
            {"type": "mrkdwn", "text": f"*RA*: {self.round_sigfigs(row.ra)} deg"}
        )
        fields['fields'].append(
            {"type": "mrkdwn", "text": f"*Declination*: {self.round_sigfigs(row.dec)} deg"}
        )
        fields['fields'].append(
            {
                "type": "mrkdwn",
                "text": f"*Peak Mag*: {self.round_sigfigs(row.peak_mag)}, {self.round_sigfigs(row.peak_phase)} days ago",
            }
        )

        attachment.append(fields)

        fields_filter = {
            "type": "section",
            "fields": []
        }
        fields_host = {
            "type": "section",
            "fields": []
        }
        for p in row.index:
            if p in (
                'tns_name', 'tns_class', 'tns_redshift',
                'ra', 'dec', 'peak_mag', 'peak_phase',
                'posted_before'
            ):
                continue
            if "host" in p:
                fields_host['fields'].append(
                    {"type": "mrkdwn", "text": f"*{self.unsnake(p)}*: {self.round_sigfigs(row[p])}"},
                )
            else:
                fields_filter['fields'].append(
                    {"type": "mrkdwn", "text": f"*{self.unsnake(p)}*: {self.round_sigfigs(row[p])}"},
                )

        attachment.append(fields_filter)
        attachment.append(fields_host)

        attachment.append(self.voting_action(row.name))

        attachment.append(generate_context_block(self._vote_df, row.name)) # this is where the votes are gonna be printed

        attachment.append(
            {
                "type": "divider"
            },
        )

        return attachment

    def post(self, channel): 
        '''
        Posts to a slack channel. If no string is provided, will use the string attribute of the object.

        Parameters
        ----------
        string : str, optional
            String to post to slack. If None, will use the string attribute of the object.
        channel : str, optional
            Channel to post to. Specific to workspace the bot token has been installed in.
        '''
        if self._meta['groupby'] is None:
            attachments = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "Today's candidates:"
                    }
                },
                {
                    "type": "divider"
                },
            ]
            for (_, row) in self._df.iterrows():
                attachments.extend(
                    self.generate_single_row_attachment(row)
                )
            send_slack_message(self._client, channel, attachments)

        else:
            groupby_col = self._meta['groupby']

            # first add header/split by groupby()
            for groupby_val in self._df[groupby_col].unique():
                attachments = [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": f"Candidates where {groupby_col} = {groupby_val}:"
                        }
                    },
                    {
                        "type": "divider"
                    },
                ]
                #if self._meta[f'overflow_{groupby_val}']:
                #    header += f"(limiting to top {self._meta['max_num']}, with >0.5 arcsec host offset)"
                for (_, row) in self._df.loc[
                    self._df[groupby_col] == groupby_val
                ].iterrows():
                    attachments.extend(
                        self.generate_single_row_attachment(row)
                    )
                send_slack_message(self._client, channel, attachments)

    def post_empty(self, channel):
        """Post message about no events.
        """
        attachments = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "No candidates for today!"
                }
            },
            {
                "type": "divider"
            },
        ]
        send_slack_message(self._client, channel, attachments)

class SlackVoteHandler:
    """Handles votes submitted for various messages.
    """
    def __init__(self):
        # Initialize the Slack App with your bot token and signing secret
        self.app = setup_app()

        # Register action listeners
        self.app.action(re.compile(r"vote\$*"))(self.handle_object_vote)

        # Set up Flask integration
        self.flask_app = Flask(__name__)
        self.handler = SlackRequestHandler(self.app)

        # Define the route to handle incoming Slack events
        self.flask_app.route("/slack/events", methods=["POST"])(self.slack_events)


    def load_votes_df(self, filter_name):
        """Load dataframe with votes.
        """
        os.makedirs(os.path.join(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(__file__),
                )
            ), "data", filter_name
        ), exist_ok=True)
        # Where to save voting history
        votes_fn = os.path.join(os.path.dirname(
            os.path.dirname(
                os.path.dirname(__file__),
            )
        ), "data", filter_name, "voting_history.csv")
        
        if os.path.exists(votes_fn):
            vote_df = pd.read_csv(votes_fn, index_col=0)
        else:
            vote_df = None

        return vote_df, votes_fn

    def handle_object_vote(self, ack, body, client):
        ack()
        user_id = body['user']['id']
        user_name = body['user']['name']
        action = body['actions'][0]
        action_value = action['value']
        antid = action['action_id'].split("$")[-1]
        filter_name = action['action_id'].split("$")[-2]

        vote_df, _ = self.load_votes_df(filter_name)

        if vote_df is not None:

            mask = (vote_df.index == antid) & (vote_df.UserID == user_id)

            if (vote_df is not None) and (len(vote_df.loc[mask]) > 0):
                client.chat_postEphemeral(
                    channel=body['channel']['id'],
                    user=user_id,
                    text="Voted previously; vote has been updated."
                )
        
        self.record_vote(action_value, antid, user_id, user_name, filter_name)
        self.update_slack_message(body, client, filter_name)


    def update_slack_message(self, body, client, filter_name):
        """From vote DF and event, update message.
        """
        action = body['actions'][0]
        obj_id = action['action_id'].split("$")[-1]
        blocks = body["message"]["blocks"]
        
        idx, _ = [x for x in enumerate(blocks) if x[1]['block_id'] == f"votes_{obj_id}"][0]

        vote_df, _ = self.load_votes_df(filter_name)
        if vote_df is None:
            return None

        new_context_block = generate_context_block(vote_df, obj_id)
        blocks[idx] = new_context_block

        # Update the Slack message
        client.chat_update(
            channel=body['channel']['id'],
            ts=body['message']['ts'],
            blocks=blocks,
            text="Fallback text"
        )

    def record_vote(self, vote, obj_id, user_id, user_name, filter_name):
        """Adds vote to dataframe with previous
        votes to avoid repeats.
        """
        vote_df, vote_fn = self.load_votes_df(filter_name)
        if vote_df is None:
            vote_df = pd.DataFrame(
                {'UserID': user_id, 'UserName': user_name, 'Response': vote},
                index=[obj_id,]
            )
            vote_df.index.name = 'Transient'

        mask = (vote_df.index == obj_id) & (vote_df.UserID == user_id)

        if len(vote_df.loc[mask]) > 0:
            vote_df.loc[mask, 'UserID'] = user_id
            vote_df.loc[mask, 'UserName'] = user_name
            vote_df.loc[obj_id, 'Response'] = vote

        else:
            new_df = pd.DataFrame(
                {'UserID': user_id, 'UserName': user_name, 'Response': vote},
                index=[obj_id,]
            )
            new_df.index.name = 'Transient'
            vote_df = pd.concat([vote_df, new_df])

        vote_df.to_csv(vote_fn)

    def slack_events(self):
        return self.handler.handle(request)

    def start(self, host="0.0.0.0", port=443):
        self.flask_app.run(host=host, port=port, ssl_context=('/root/cert.pem', '/root/key.pem'))

if __name__ == "__main__":
    slack_vote_handler = SlackVoteHandler()
    slack_vote_handler.start()
