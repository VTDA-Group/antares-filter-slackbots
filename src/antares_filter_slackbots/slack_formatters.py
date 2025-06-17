import numpy as np
import pandas as pd
import os
import re
import requests

from astropy.table import Table
from flask import Flask, request
from slack_bolt.adapter.flask import SlackRequestHandler

from antares_filter_slackbots.slack_requests import (
    send_slack_message,
    setup_app,
    setup_client
)

def geturl(ra, dec, size=240, output_size=None, filters="grizy", format="jpg", color=False, type='stack'):
    """Get the URL for images in the table. Taken from astro_ghost.
    """  
    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    url_table = ("{service}?ra={ra}&dec={dec}&size={size}&format=fits"
           "&filters={filters}&type={type}").format(**locals())
    table = Table.read(url_table, format='ascii')
    url = ("https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
           "ra={ra}&dec={dec}&size={size}&format={format}").format(**locals())
    if output_size:
        url = url + "&output_size={}".format(output_size)

    # sort filters from red to blue
    flist = ["yzirg".find(x) for x in table['filter']]
    table = table[np.argsort(flist)]
    if color:
        if len(table) > 3:
            # pick 3 filters
            table = table[[0,len(table)//2,len(table)-1]]
        for i, param in enumerate(["red","green","blue"]):
            url = url + "&{}={}".format(param, table['filename'][i])
    else:
        urlbase = url + "&red="
        url = []
        for filename in table['filename']:
            url.append(urlbase+filename)
    return url


def is_link_accessible(url):
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        return response.status_code < 400
    except requests.RequestException as e:
        # Could not access the link
        print(f"Error: {e}")
        return False
    
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
        self._tns_url_base = 'https://www.wis-tns.org/object/'
        self._url_base = 'https://antares.noirlab.edu/loci/'
        self._df = loci_df
        self._meta = filt_meta

        # Where to save voting history
        self._votes_fn = os.path.join(save_prefix, "voting_history.csv")
        self.filter_name = save_prefix.split("/")[-1]
        
        if os.path.exists(self._votes_fn):
            self._vote_df = pd.read_csv(self._votes_fn, index_col=0)
        else:
            self._vote_df = None

            
    def round_sigfigs(self, x, sig=3):
        """Round numerics to sig significant figures.
        """
        if isinstance(x, (int, float, np.number)):  # Check for numeric types
            if isinstance(x, bool):
                return x
            elif (x == 0):  # Handle zero separately to avoid log10 issues
                return x
            elif np.isnan(x):
                return '---'
            else:
                return round(x, -int(np.floor(np.log10(abs(x)))) + (sig - 1))
        elif isinstance(x, str) and x in ['', 'nan']:
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
            return geturl(row.ra, row.dec, color=True)
        
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
        if self._vote_df is not None:
            sub_df = self._vote_df.loc[self._vote_df.index == row.name]

            if len(sub_df.loc[sub_df.Response == 'downvote']) > 1:
                return []

        title_link = self._url_base + row.name
        
        if self.round_sigfigs(row.tns_name) != '---':
            tns_url = self._tns_url_base + row.tns_name
            if not row.name[:4].isnumeric(): # no repeating titles
                title_str = f":collision: <{tns_url}|*{row.tns_name}*> | <{title_link}|*{row.name}*>"
            else:
                title_str = f":collision: <{tns_url}|*{row.tns_name}*>"
                
        elif is_link_accessible(title_link):
            title_str = f":collision: <{title_link}|*{row.name}*>"
        else:
            title_str = f":collision: *{row.name}*"
            
        if ('yse_pz' in row.index) and (row.yse_pz is not None):
            title_str += f" | {row.yse_pz}"
        title_str += " :collision:"

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
                    "text": title_str
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

        fields_overview = {
            "type": "section",
            "fields": []
        }
        fields_tns = {
            "type": "section",
            "fields": []
        }

        fields_overview['fields'].append(
            {"type": "mrkdwn", "text": f"*RA*: {self.round_sigfigs(row.ra)} deg"}
        )
        fields_overview['fields'].append(
            {"type": "mrkdwn", "text": f"*Declination*: {self.round_sigfigs(row.dec)} deg"}
        )
        fields_overview['fields'].append(
            {
                "type": "mrkdwn",
                "text": f"*Peak Mag*: {self.round_sigfigs(row.peak_mag)}, {self.round_sigfigs(row.peak_phase)} days ago",
            }
        )
        abs_mag = self.round_sigfigs(row.peak_abs_mag)
        
        if abs_mag != '---':
            fields_overview['fields'].append(
                {
                    "type": "mrkdwn",
                    "text": f"*Peak Abs Mag*: {abs_mag}"
                }
            )
            
        if self.round_sigfigs(row.yse_field) != '---':
             fields_overview['fields'].append(
                {
                    "type": "mrkdwn",
                    "text": f"*YSE Field*: {self.round_sigfigs(row.yse_field)}"
                }
            )

        attachment.append(fields_overview)

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
                'tns_name', 'ra', 'dec', 'peak_mag', 'peak_abs_mag', 'peak_phase',
                'posted_before', 'best_redshift', 'yse_pz', 'yse_field'
            ):
                continue
            value = self.round_sigfigs(row[p])
                
            if value == '---': # just exclude for now, can always turn this off
                continue
                
            if ("host" in p) or (p == 'nuclear'):
                fields_host['fields'].append(
                    {"type": "mrkdwn", "text": f"*{self.unsnake(p)}*: {self.round_sigfigs(row[p])}"},
                )
            elif ("tns" in p):
                fields_tns['fields'].append(
                    {"type": "mrkdwn", "text": f"*{self.unsnake(p)}*: {self.round_sigfigs(row[p])}"},
                )
            else:
                fields_filter['fields'].append(
                    {"type": "mrkdwn", "text": f"*{self.unsnake(p)}*: {self.round_sigfigs(row[p])}"},
                )
        
        if len(fields_tns['fields']) > 0:
            attachment.append(fields_tns)
        if len(fields_host['fields']) > 0:
            if len(fields_host['fields']) > 8:
                fields_host1 = {
                    "type": "section",
                    "fields": fields_host['fields'][:8]
                }
                fields_host2 = {
                    "type": "section",
                    "fields": fields_host['fields'][8:]
                }
                attachment.append(fields_host1)
                attachment.append(fields_host2)
            else:
                attachment.append(fields_host)
                
        if len(fields_filter['fields']) > 0:
            if len(fields_filter['fields']) > 8:
                fields_filter1 = {
                    "type": "section",
                    "fields": fields_filter['fields'][:8]
                }
                fields_filter2 = {
                    "type": "section",
                    "fields": fields_filter['fields'][8:]
                }
                attachment.append(fields_filter1)
                attachment.append(fields_filter2)
            else:
                attachment.append(fields_filter)
        
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
            send_slack_message(self._client, channel, attachments, "Today's Candidates")
            
            for (_, row) in self._df.iterrows():
                attachments = self.generate_single_row_attachment(row)
                if len(attachments) > 0:
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
                send_slack_message(self._client, channel, attachments)
                
                for (_, row) in self._df.loc[
                    self._df[groupby_col] == groupby_val
                ].iterrows():
                    attachments = self.generate_single_row_attachment(row)
                    if len(attachments) > 0:
                        send_slack_message(self._client, channel, attachments, row.name)

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
        timestamp = body['message']['ts']

        vote_df, _ = self.load_votes_df(filter_name)

        if vote_df is not None:

            mask = (vote_df.index == antid) & (vote_df.UserID == user_id)

            if (vote_df is not None) and (len(vote_df.loc[mask]) > 0):
                client.chat_postEphemeral(
                    channel=body['channel']['id'],
                    user=user_id,
                    text="Voted previously; vote has been updated."
                )
        
        self.record_vote(action_value, antid, user_id, user_name, filter_name, timestamp)
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
            text=f"Source {obj_id} has been updated."
        )

    def record_vote(self, vote, obj_id, user_id, user_name, filter_name, timestamp):
        """Adds vote to dataframe with previous
        votes to avoid repeats.
        """
        vote_df, vote_fn = self.load_votes_df(filter_name)
        if vote_df is None:
            vote_df = pd.DataFrame(
                {'UserID': user_id, 'UserName': user_name, 'Response': vote, 'Timestamp': timestamp},
                index=[obj_id,]
            )
            vote_df.index.name = 'Transient'

        mask = (vote_df.index == obj_id) & (vote_df.UserID == user_id) & (vote_df.Response == vote)

        if len(vote_df.loc[mask]) > 0:
            vote_df.loc[mask, 'Timestamp'] = timestamp

        else:
            new_df = pd.DataFrame(
                {'UserID': user_id, 'UserName': user_name, 'Response': vote, 'Timestamp': timestamp},
                index=[obj_id,]
            )
            new_df.index.name = 'Transient'
            vote_df = pd.concat([vote_df, new_df])

        vote_df.to_csv(vote_fn)

        
    def slack_events(self):
        return self.handler.handle(request)

    def start(self, host="0.0.0.0", port=443):
        self.flask_app.run(host=host, port=port, ssl_context=('/root/cert.pem', '/root/key.pem'))
        
        
        
class YSESlackPoster(SlackPoster):
    def __init__(self, loci_df, filt_meta, save_prefix):
        '''
        Reformats YSE information for Slack posting.

        Parameters
        ----------
        loci_df: pd.DataFrame
            Stores extracted locus features to display with Slack bot
        '''
        super().__init__(loci_df, filt_meta, save_prefix)
        self._url_base = 'https://ziggy.ucolick.org/yse/transient_detail/'
            
            

if __name__ == "__main__":
    slack_vote_handler = SlackVoteHandler()
    slack_vote_handler.start()