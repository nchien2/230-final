import SoccerNet
import config
from SoccerNet.Downloader import SoccerNetDownloader
mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="SoccerNet/")

mySoccerNetDownloader.password = config.password  # Password for videos
mySoccerNetDownloader.downloadGames(files=["2_224p.mkv"], split=[
                                    "valid", "test", "challenge"])
mySoccerNetDownloader.downloadDataTask(task="caption-2023", split=["test"])
