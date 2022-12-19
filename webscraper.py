import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


class Dataset:
    """Class for scraping web data, and storing Job objects"""

    def __init__(self, **kwargs):
        self._jobs = []
        self._skills = []
        self.initialize(**kwargs)

    def initialize(self, **kwargs):
        URLS = kwargs.get('URLS', None)
        load_file = kwargs.get('load_file', None)
        if URLS:
            print('Iterating through each url and extracting job postings...')
            for URL in tqdm(URLS):
                page = requests.get(URL)
                soup = BeautifulSoup(page.text, "html.parser")
                self._jobs += self.batch_extract_job_title_and_link(soup)
            print('Total number of jobs found: %d' % len(self._jobs))
        elif load_file:
            import pandas as pd
            self._skills = pd.read_csv(load_file).dropna()
            self._skills = self._skills[self._skills.columns[0]].to_list()

    def save_xml(self):
        pass

    def save_csv(self, file_name="out.csv"):
        import pandas as pd
        if not self._skills:
            self.populate_skills_dict()
        pd.DataFrame(self._skills).to_csv(file_name, index=False)

    def batch_extract_job_title_and_link(self, soup_object):
        jobs = []
        for div in soup_object.find_all(name="div", attrs={"class": "row"}):
            for a in div.find_all(name="a", attrs={"data-tn-element": "jobTitle"}):
                if 'rc/clk' in a["href"]:
                    jobs.append(Job(title=a["title"], url='https://www.indeed.com' + a["href"]))
        return jobs

    def populate_skills_dict(self):
        print('Iterating through each post and extracting skills...\n')
        for job in tqdm(self._jobs):
            self._skills += job.find_skills_from_links()

    def preprocess_skills_dict(self):
        import re
        for i in range(len(self._skills)):
            self._skills[i] = re.sub(r'[,\.!?]', '', self._skills[i]).lower()


class Job:
    """Class containing metadata about a job, and a method for parsing job descriptions"""

    def __init__(self, **kwargs):
        self.title = kwargs.get('title', '')
        self.URL = kwargs.get('url', '')
        self.skills = []

    def find_skills_from_links(self):
        page = BeautifulSoup(requests.get(self.URL).text, 'html.parser')
        bullets = page.find_all('li', attrs={'class': None})
        skills = []
        for s in bullets:
            skills.append(s.text.strip())
        self.skills = skills
        return skills
