from NativeBayes.nativeBayes import *
from NativeBayes.rss import  *
import feedparser
# print(spamTest())
feedAddr1='http://newyork.craigslist.org/stp/index.rss'
feedAddr2='http://sfbay.craigslist.org/stp/index.rss'
feedAddr3='http://sfbay.craigslist.org/search/res?format=rss'
feedAddr4='http://newyork.craigslist.org/search/res?format=rss'
print(localWords(feedAddr3,feedAddr4))