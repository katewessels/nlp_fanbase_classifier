import requests
import json
import time
import pandas as pd
import numpy as np


url = "https://api.pushshift.io/reddit/search/submission"

# params = {"subreddit": "gratefuldead"}
# submissions = requests.get(url, params = params)

# submissions =  submissions.json()
# submissions_dict = json.loads(submissions)

#get only submissions before the last from the last query
# last_submission_time = submissions.json()["data"][-1]["created_utc"]
# params = {"subreddit" : "gratefuldead", "before" : last_submission_time}
# submissions = requests.get(url, params = params)



def crawl_page(subreddit: str, last_page = None):
  """Crawl a page of results from a given subreddit.

  :param subreddit: The subreddit to crawl.
  :param last_page: The last downloaded page.

  :return: A page or results.
  """
  params = {"subreddit": subreddit, "size": 500, "sort": "desc", "sort_type": "created_utc"}
  if last_page is not None:
    if len(last_page) > 0:
      # resume from where we left at the last page
      params["before"] = last_page[-1]["created_utc"]
    else:
      # the last page was empty, we are past the last page
      return []
  results = requests.get(url, params)
  if not results.ok:
    # something wrong happened
    raise Exception("Server returned status code {}".format(results.status_code))
  return results.json()["data"]


def crawl_subreddit(subreddit, max_submissions = 2000):
  """
  Crawl submissions from a subreddit.

  :param subreddit: The subreddit to crawl.
  :param max_submissions: The maximum number of submissions to download.

  :return: A list of submissions.
  """
  submissions = []
  last_page = None
  while last_page != [] and len(submissions) < max_submissions:
    last_page = crawl_page(subreddit, last_page)
    submissions += last_page
    time.sleep(3)
  return submissions[:max_submissions]

def make_dataframe(list_of_submissions):
    '''
    take list of submissions (output from crawl_subreddit())
    and transform into a dataframe with columns:
    ['id', 'author', 'title', 'score',
    'url', 'full_link', 'num_comments', 'created_utc', 'subreddit']

    also saves df to csv in data folder
    '''
    posts = []
    for i in range(len(list_of_submissions)):
        posts.append([list_of_submissions[i]['id'],
                    list_of_submissions[i]['author'],
                    list_of_submissions[i]['title'],
                    list_of_submissions[i]['score'],
                    list_of_submissions[i]['url'],
                    list_of_submissions[i]['full_link'],
                    list_of_submissions[i]['num_comments'],
                    list_of_submissions[i]['created_utc'],
                    list_of_submissions[i]['subreddit']
        ])
    posts_df = pd.DataFrame(posts, columns=['id', 'author', 'title', 'score',\
                   'url', 'full_link', 'num_comments', 'created_utc', 'subreddit'])

    return posts_df

def get_submissions_to_csv(subreddit, max_submissions):
    submissions = crawl_subreddit(subreddit, max_submissions=max_submissions)
    df = make_dataframe(submissions)
    df.to_csv(f'data/{subreddit}.csv', index=None)


if __name__ == "__main__":

    #to see how many total submissions in a subreddit:
    #gd
    # requests.get(url, params = {"subreddit": "gratefuldead", "size": 0, "aggs" : "subreddit"}).json()["aggs"]

    #tests
    # gd_submissions: a list of submissions(dictionary)
    # gd_submissions = crawl_subreddit('gratefuldead', max_submissions=20000)

    #scrape data
    #get data to csv files
    # get_submissions_to_csv('gratefuldead', max_submissions=40000)
    # get_submissions_to_csv('pinkfloyd', max_submissions=40000)
    # get_submissions_to_csv('phish', max_submissions=40000)
    # get_submissions_to_csv('beatles', max_submissions=40000)

    #get csv files to pandas dataframes
    df_gd = pd.read_csv('data/gratefuldead.csv')
    df_pf = pd.read_csv('data/pinkfloyd.csv')
    df_phish = pd.read_csv('data/phish.csv')
    df_beatles = pd.read_csv('data/beatles.csv')


    #pd.set_option('display.max_columns')

    #merge/concat dataframes into single dataframe
    df = pd.concat([df_gd, df_pf, df_phish, df_beatles], join  = 'outer', axis = 0, ignore_index = True)