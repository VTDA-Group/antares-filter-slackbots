# Main functions to create HTML file
import os
import numpy as np
from datetime import datetime
import requests
from astropy.table import Table

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

def ps1_pic(row):
    """Retrive PS1 image url of entry."""
    if row.dec > -30:
        return geturl(row.ra, row.dec, color=True)

    return None

def unsnake(col):
    """Convert column names from snake case
    to normal spacing/capitalization.
    """
    c_split = col.split("_")
    c_capitalized = [word.capitalize() for word in c_split]
    c_reformat = ' '.join(c_capitalized)
    return c_reformat
    
    
def round_sigfigs(x, sig=3):
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
    
    
def generate_table_columns(df):
    """Generate table columns for slackbot"""
    column_str = '<th>Event Name</th>'
    column_str += '\n<th>PS1 Stamp</th>'
    column_str += '\n<th>RA</th>\n<th>Declination</th>'
    column_str += '\n<th>Peak Mag</th>\n<th>Peak Abs Mag</th>\n<th>Days Since Peak</th>'
    column_str += '\n<th>YSE Field</th>'

    host_fields = []
    tns_fields = []
    other_fields = []
    
    for p in df.columns:
        if p in (
            'tns_name', 'ra', 'dec', 'peak_mag', 'peak_abs_mag', 'peak_phase',
            'posted_before', 'best_redshift', 'yse_pz', 'yse_field'
        ):
            continue

        if ("host" in p) or (p == 'nuclear'):
            host_fields.append(unsnake(p))
        elif ("tns" in p):
            tns_fields.append(unsnake(p))
        else:
            other_fields.append(unsnake(p))
            
    host_str = ''.join([f'\n<th>{p}</th>' for p in host_fields])
    tns_str = ''.join([f'\n<th>{p}</th>' for p in tns_fields])
    other_str = ''.join([f'\n<th>{p}</th>' for p in other_fields])
    
    column_str += tns_str + other_str + host_str
    return column_str


def generate_table_rows(df, vote_df, url_base, tns_url_base):
    """Generate table rows for slackbot"""
    # Create the table rows
    rows = ''
    for (_, row) in df.iterrows():
        if vote_df is not None:
            sub_df = vote_df.loc[vote_df.index == row.name]

            if len(sub_df.loc[sub_df.Response == 'downvote']) > 1:
                continue

        title_link = url_base + row.name
        
        if round_sigfigs(row.tns_name) != '---':
            tns_url = tns_url_base + row.tns_name
            if not row.name[:4].isnumeric(): # no repeating titles
                title_str = f'<a href="{tns_url}">{row.tns_name}</a>\n<a href="{title_link}">{row.name}</a>'
            else:
                title_str = f'<a href="{tns_url}">{row.tns_name}</a>'
                
        elif is_link_accessible(title_link):
            title_str = f'<a href="{title_link}">{row.name}</a>'
        else:
            title_str = f"{row.name}"
            
        if ('yse_pz' in row.index) and (row.yse_pz is not None):
            title_str += f'''\n<a href="{row.yse_pz.split('|')[0][1:]}">YSE-PZ</a>'''

        rows += f'<tr><td>{title_str}</td>'
        try:
            img_html = f'<img src="{ps1_pic(row)}" width="100" height="100">'
        except:
            img_html = 'NA'
        
        rows += f'<td>{img_html}</td>'
        rows += f'<td>{round_sigfigs(row.ra)}</td>'
        rows += f'<td>{round_sigfigs(row.dec)}</td>'
        rows += f'<td>{round_sigfigs(row.peak_mag)}</td>'
        rows += f'<td>{round_sigfigs(row.peak_abs_mag)}</td>'
        rows += f'<td>{round_sigfigs(row.peak_phase)}</td>'
        rows += f'<td>{round_sigfigs(row.yse_field)}</td>'
        
        host_rows = ''
        other_rows = ''
        tns_rows = ''

        for p in row.index:
            if p in (
                'tns_name', 'ra', 'dec', 'peak_mag', 'peak_abs_mag', 'peak_phase',
                'posted_before', 'best_redshift', 'yse_pz', 'yse_field'
            ):
                continue
                
            if isinstance(row[p], str) and ('https' in row[p]):
                value = f'<a href="{row[p]}">Click Here</a>'
            else:
                value = round_sigfigs(row[p])
            
            if ("host" in p) or (p == 'nuclear'):
                host_rows += f'\n<td>{value}</td>'
            elif ("tns" in p):
                tns_rows += f'\n<td>{value}</td>'
            else:
                other_rows += f'\n<td>{value}</td>'
        
        rows += tns_rows
        rows += other_rows
        rows += host_rows
        rows += '</tr>'
    return rows


def create_html_tab(df, poster):
    """Create HTML tab."""
    prefix = poster.filter_name
    save_folder = os.path.dirname(poster._votes_fn)
    
    current_date = datetime.now().strftime("%Y-%m-%d")
    github_link = 'https://github.com/VTDA-Group/antares-filter-slackbots.git'

    html_template = '''
        <html>
            <head>
                <title>Candidates for {filt} filter on {date}</title>
                <style>
                    body {{
                        background-color: #fdf6f0;
                        color: #2e1a1a;
                        font-family: 'Georgia', serif;
                        margin: 0;
                        padding: 20px;
                    }}
                    h1 {{
                        text-align: center;
                        color: #8a100b;
                        font-size: 2.8em;
                        margin-bottom: 30px;
                    }}
                    table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin-top: 20px;
                        box-shadow: 0 0 10px rgba(138, 16, 11, 0.2);
                    }}
                    th, td {{
                        padding: 14px;
                        text-align: center;
                    }}
                    th {{
                        background-color: #8a100b;
                        color: #fffaf5;
                        font-size: 1.1em;
                        text-transform: uppercase;
                        border-bottom: 2px solid #660000;
                    }}
                    td {{
                        background-color: #fffaf5;
                        border-bottom: 1px solid #ddd;
                    }}
                    tr:hover {{
                        background-color: #fbe9e7;
                        transition: background-color 0.3s;
                    }}
                    a {{
                        color: #8a100b;
                        text-decoration: none;
                        font-weight: bold;
                    }}
                    a:hover {{
                        text-decoration: underline;
                    }}
                    img {{
                        border-radius: 8px;
                        transition: transform 0.3s;
                    }}
                    img:hover {{
                        transform: scale(1.05);
                    }}
                    footer {{
                        margin-top: 40px;
                        text-align: center;
                        font-size: 0.9em;
                        color: #555;
                    }}
                </style>
            </head>
            <body>
                <h1>Candidates for {filt} filter on {date}</h1>
                <table>
                    <tr>
                        {columns}
                    </tr>
                    {rows}
                </table>
                <footer>
                    <p>Code can be found here: <a href="{github}">Slackbot GitHub</a></p>
                </footer>
            </body>
        </html>
        '''

    # Render the HTML content
    columns = generate_table_columns(df)
    rows = generate_table_rows(df, poster._vote_df, poster._url_base, poster._tns_url_base)
    html_content = html_template.format(
        rows=rows, columns=columns, date=current_date,
        github=github_link, filt=unsnake(prefix)
    )

    # Write the HTML content to a file
    save_path = os.path.join(save_folder, 'table.html')
    with open(save_path, 'w') as f:
        f.write(html_content)
