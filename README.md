# antares-filter-slackbots

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)

This repository connects with the ANTARES and ALeRCE alert brokers, along with ATLAS, TNS, and YSE-PZ to prioritize photometric alerts for rapid follow-up. These alerts of interest are then sent to the YSE community via Slackbot messages. This repo can easily be augmented to assist other research collaborations.

Different science goals are accommodated through the creation of *filters*, which take in timeseries data and a dictionary of meta-information and adds properties to the meta-dictionary. These properties can then be used to prioritize and filter newly observed objects.

## Environment
Dependencies will be installed with the package using `pip install -e .` from the root directory. However, you also need to manually update scikit-learn after using `pip install --upgrade scikit-learn`. To use the TNS and ATLAS retrievers, you must also make sure the relevant credentials are in your .bashrc file (see SNAPI documentation for details).

## Running the Slackbot filters
The script is run with the *run()* function in the *run.py* file. For it to actually run properly, you will need to update your *auth.py* file with the proper credentials. These credentials allow certain data retrievers to obtain data and also allow the codebase to interact with your Slack Workspace through the Slack API (https://api.slack.com/). You first must create a App associated with your Slack Workspace through the website, and then generate the following (from the left sidebar):
* Under "Basic Information", copy the "Signing Secret" to a "signing_secret" variable in auth.py.
* Under "OAuth & Permissions", copy the "Bot User OAuth Token" to a "toku" variable in auth.py.
* You may need to also add "login" and "password" variables if you added a username and password requirement for your Slackbot on the API site.

To run specific data retrievers, you may need the following additional variables in `auth.py`:
* To use TNSRetriever or ATLASRetriever, you must add the `tns_id`, `tns_name`, and `tns_key` associated with your bot (https://www.wis-tns.org/bots) to `auth.py`.
* To use YSERetriever, you must add `login_ysepz` and `password_ysepz` variables with your YSE-PZ login credentials.

To add a filter, you want to add a new `RankingFilter` object in `all_current_filters()`. The arguments for `RankingFilter` initialization are (in order):
* the name of the filter
* retriever object to use
* filter to use (from filters/ folder)
* Slack channel to post alerts
* property used to order and prioritize alerts
* (optional) pre-filter properties (used in ANTARES queries to narrow down loci right off the bat)
* (optional) save properties: additional properties to post in the alert
* (optional) post-filter tags: tags to filter events by after applying filter itself
* (optional) post-filter properties: properties to filter events by after applying filter
* (optional) groupby filters: group returned alerts by value of this property
* (optional) ascending: prioritize events with the smallest rank property instead of the largest

## Data Retrievers
A variety of Retriever classes are available, to obtain photometric data from different sources. The default is ANTARESRetriever, which returns ZTF public survey data through `antares-client` (https://nsf-noirlab.gitlab.io/csdc/antares/client/index.html), and adds any ZTF forced photometry available through ALeRCE (https://alerce.science/). RelaxedANTARESRetriever is similar but with a lower bar for light curve quality and fewer catalog checks. YSERetriever obtains data from the YSE-PZ database, while TNSRetriever and ATLASRetriever both check TNS for new transients and then query ATLAS for forced photometry (TNSRetriver has more relaxed quality checks).

## Quality Checks
All quality check functions are found in a dedicated file. Retrievers each have a `self.quality_check()` function which calls one of the functions in this file (making it easy to make cuts stricter or more relaxed).

## Host Galaxy Association
While sometimes spectroscopic redshifts are available on TNS, we often want to follow-up on new transients without spectra. Therefore, we associate events with their most likely host galaxy using Prost (https://github.com/alexandergagliano/Prost). Galaxies from the DECaLs and GLADE catalogs are used for this cross-matching. We then save basic host information (including redshift if available) to the metadata associated with each event, where this information can be used by downstream filters. We also check whether the transient is likely nuclear using iinuclear (https://github.com/gmzsebastian/iinuclear).

## Filters
All filters are found in the `src/filters` folder, and associated data is found in `data/filters`. Filters inherit from the new ANTARES DevKit2 (https://gitlab.com/nsf-noirlab/csdc/antares/devkit2_poc) `BaseFilter` class, with a required `setup()` and `_run()` function. The only difference is our `_run()` function is modified to take in an event dictionary and a time series dataframe instead of a Locus object. Our filters are structured similarly to those on ANTARES so that integrating into the official ANTARES filter system is more straightforward.

The current existing filters are:
* superphot-plus: to classify supernovae among five subtypes.
* precursor-emission: to search proactively for precursor emission before SNe explode. Also used to mark nuclear transients.
* shapley: generates the SHAP plots for events marked with high LAISS anomaly scores.

## Voting on Slackbot Messages
The Slackbots return promising alerts based on the provided filter criteria. Due to pre-processing or edge cases in filter behavior, however, some alerts may actually be more scientifically useful than others. Therefore, we have integrated a way for Slack users to vote on the alerts returned by the Slackbots. For this functionality to work, you must have a live URL that can accept POST HTTPS requests. On the Slack API website, under "Interactivity & Shortcuts" for your app, change "Request URL" to this URL. You then want to deploy a gunicorn instance with the voting handler through that URL. For example, I connected a local server to a live URL through ngrok (https://ngrok.com/), and then ran:
```
nohup gunicorn -b 0.0.0.0:(PORT) --certfile=/root/cert.pem --keyfile=/root/key.pem wsgi:flask_app &
```
from within `antares-filter-slackbots/src/antares_filter_slackbots`. You will need SSL cert and key files to use this method. Once this is working, you will be able to click the voting buttons under each Slackbot alert, and will see others' votes as well. These votes are saved within the data folders for each filter.

See more info on [superphot-plus](https://github.com/lincc-frameworks/superphot-plus)

## Automating the Slackbot Workflow
This codebase is optimized for efficiency, such that it can be run automatically on a daily basis and be done in less than 30 minutes (with reasonable filter constraints). Therefore, I highly recommend you automate your workflow by integrating the run call into a crontab script. Here is what my script looks like (note that `#!/bin/bash` must be added when running on a distributed cluster using *scrontab*):
```
cd (repo location)/antares-filter-slackbots
git pull
cd (repo location)
(env_path)/bin/python (repo location/antares-filter-slackbots/src/antares_filter_slackbots/run.py
```
Then with `crontab -e` simply call the file with these lines. The `git pull` is important to pull any votes from the previous day, since events with multiple downvotes are no longer posted.

If voting is set up, I also have a cron job set up on the voting server side:
```
cd (repo location)/antares-filter-slackbots
git add *
git commit -m "Daily vote push"
git push
```
This actually pushes the votes to the GitHub repository. I suggest having this set to run **before** the cronjob that posts new alerts.

The cronjob line itself is:
```
#SCRON -t 0-02:00 --mem 16000 -o /n/holylabs/LABS/avillar_lab/Lab/slackbots.out -e /n/holylabs/LABS/avillar_lab/Lab/slackbots.err --partition test
0 7 * * * /n/holylabs/LABS/avillar_lab/Lab/ztf_slackbot_cron.sh
```
