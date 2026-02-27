import requests

def search_yse_for_transient(event_dict, auth):
    """Event dict must have:
    - tns_name
    - ra
    - dec
    """
    tns_name = event_dict['tns_name']
    if (tns_name is not None) and str(tns_name)[:4].isnumeric():
        yse_search_url = f"https://ziggy.ucolick.org/yse/api/transients/?name={tns_name}"
    else:
        ra_lower = event_dict['ra'] - 0.0001
        ra_upper = event_dict['ra'] + 0.0001
        dec_lower = event_dict['dec'] - 0.0001
        dec_upper = event_dict['dec'] + 0.0001
        yse_search_url = f"https://ziggy.ucolick.org/yse/api/transients/?ra_gte={ra_lower}"
        yse_search_url += f"&ra_lte={ra_upper}&dec_gte={dec_lower}&dec_lte={dec_upper}"
        
    yse_results = requests.get(yse_search_url, auth=auth).json()['results']

    if len(yse_results) > 0:
        yse_result = yse_results[0]
        return yse_result
    
    return None


def add_tag_to_yse(event_dict, tag_path, auth):
    """Add tag to YSE."""
    yse_result = search_yse_for_transient(event_dict, auth)

    if yse_result is None:
        return None

    yse_result['tags'].append(tag_path)
    yse_result['tags'] = list(set(yse_result['tags']))
    url = yse_result['url']
    requests.put(
        url,
        json=yse_result,
        auth=auth
    )
    return yse_result
    

                
def make_new_yse_tag(tag_name, color_id, auth):
    """
    Add a new tag name to YSE-PZ

    Warning: doesn't chekc if the tag already exists

    args:
        tag_name (str): the name of the tag you want to add
        color_id (int): the id of the color you want to use for the tag from YSE API's webappcolors
    """

    n_tags = requests.get('https://ziggy.ucolick.org/yse/api/transienttags/', auth=auth).json()['count']

    new_tag_info = {
            "url": f"https://ziggy.ucolick.org/yse/api/transienttags/{n_tags+1}/",
            "color": f"https://ziggy.ucolick.org/yse/api/webappcolors/{color_id}/", # slackbot is 4
            "name": f"{tag_name}",
        }

    r = requests.post('https://ziggy.ucolick.org/yse/api/transienttags/', 
                      json=new_tag_info, auth=auth)
    return r
    