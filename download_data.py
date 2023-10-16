import argparse
import os
import requests
import zipfile
import xml.etree.ElementTree as ET
from glob import glob
import json
from pytube import YouTube
from pytube.exceptions import AgeRestrictedError

from utils.logger import LoggerManager

# intialise logger
logger = LoggerManager.get_logger(log_file='log_files/download_data.log')


def make_dir(path):
    """
    Create a new directory at the specified path if it does not exist.
    
    Parameters
    ----------
    path : str
        The path of the directory to be created.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def download_and_unzip(url, extract_to='.'):
    """
    Download a zip file from a URL and extract its contents.
    
    Downloads a zip file from the provided URL, saves it locally as 'temp.zip',
    extracts its contents to a specified directory, and then removes the temporary zip file.
    Log information is generated for download success and extraction, as well as for failed download attempts.
    
    Parameters
    ----------
    url : str
        The URL pointing to the zip file to be downloaded.
    extract_to : str, optional
        Path where the zip content should be extracted to. Defaults to the current directory.
    """
    # Send a HTTP request to the URL
    response = requests.get(url)
    
    # Check if request was successful (HTTP Status Code 200)
    if response.status_code == 200:
        # Write the content of the response to a local zip file
        with open('temp.zip', 'wb') as f:
            f.write(response.content)
        
        # Create a ZipFile object
        with zipfile.ZipFile('temp.zip', 'r') as zip_ref:
            # Extract all contents of the zip file to the specified directory
            zip_ref.extractall(extract_to)
        
        # Clean up by removing the temporary zip file
        os.remove('temp.zip')
        logger.info(f"The zip file was downloaded and extracted to {extract_to}")
    else:
        logger.info(f"Failed to download the file. Status code: {response.status_code}")


def xml_to_dict(file_path):
    """
    Parse an XML file and convert it to a dictionary.
    
    This function reads an XML file from the provided file path, parses its contents, 
    and converts them into a dictionary with a predefined structure. The XML should contain
    video data, including an id, keyword, url, and other attributes. It will log and return 
    None if an exception occurs during the parsing or file reading process.
    
    Parameters
    ----------
    file_path : str
        Path to the XML file that needs to be parsed and converted to a dictionary.

    Returns
    -------
    video_dict : dict
        The parsed XML data as a dictionary, adhering to a predefined structure 
        related to video data, or None if an error occurred during the process.
    """
    try:
        # Parse the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Construct the dictionary
        video_dict = {
            "id": root.attrib.get("id"),
            "keyword": root.attrib.get("keyword"),
            "url": root.find("url").text,
            "height": int(root.find("height").text),
            "width": int(root.find("width").text),
            "frames": int(root.find("frames").text),
            "persons": int(root.find("persons").text),
            "duration": root.find("duration").text,
            "conversation": root.find("conversation").text == "yes",
            "behaviours": {
                "count": int(root.find("behaviours").attrib.get("count")),
                "id": root.find("behaviours").attrib.get("id"),
                "list": [
                    {
                        "id": b.attrib.get("id"),
                        "time": b.find("time").text,
                        "bodypart": b.find("bodypart").text,
                        "category": b.find("category").text,
                        "intensity": b.find("intensity").text,
                        "modality": b.find("modality").text,
                    }
                    for b in root.find("behaviours").findall("behaviour")
                ],
            },
        }
        return video_dict
    
    except ET.ParseError:
        logger.info("Error: The XML file is malformed.")
        return None
    except FileNotFoundError:
        logger.info(f"Error: No such file or directory: '{file_path}'")
        return None
    except Exception as e:
        logger.info(f"An error occurred: {str(e)}")
        return None


def download_youtube_video(url, filename=None, path='.'):
    """
    Download a YouTube video in the highest available resolution.

    Downloads a YouTube video given a URL, saving it to a specified path and
    under a specified filename (if provided). If the video is age-restricted,
    it will not be downloaded, and the function will return its URL and filename
    within a dictionary. In the case of other exceptions or if the download 
    is successful, it returns None. Logs will inform about the downloading process,
    successful download location, age restriction, or any errors encountered.
    
    Parameters
    ----------
    url : str
        YouTube video URL.
    filename : str, optional
        Desired filename without extension; default is YouTube title.
    path : str, optional
        Path where to save the video; default is current directory.

    Returns
    -------
    dict or None
        A dict of filename as key and url as value if video is age restricted.
        None if the video was successfully downloaded or if an error occurred.
    """
    try:
        # Create a YouTube object
        yt = YouTube(url)
        
        # Add extension
        filename_with_extension = f"{filename}.mp4" if filename else f"{yt.title}.mp4"

        # Get the stream with the highest resolution
        stream = yt.streams.get_highest_resolution()

        # Download the video
        logger.info(f"Downloading: {yt.title}")
        stream.download(output_path=path, filename=filename_with_extension)
        logger.info(f"Download saved to {os.path.join(path, filename_with_extension)}")

    except AgeRestrictedError:
        # If video is age restricted, return the url and name
        logger.info(f"Video {yt.title} is age-restricted and will not be downloaded.")
        return {filename_with_extension : url}

    except Exception as e:
        logger.info(f"An error occurred: {str(e)}")

    return None


if __name__ == "__main__":

    # parser to get data path
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data', help='path to store dataset')
    args = parser.parse_args()


    # create data path
    data_path = args.data_path
    make_dir(data_path)

    # path to download annotations data
    ssbd_path = os.path.join(data_path, 'ssbd')
    make_dir(ssbd_path)

    # link to ssbd dataset
    url = "https://rolandgoecke.files.wordpress.com/2019/11/ssbd-release.zip"

    # download the ssbd data and unzip it
    download_and_unzip(url, ssbd_path)

    # identify all the xml annotations
    xml_files = glob(os.path.join(data_path, 'ssbd', 'Annotations', '*.xml'))


    # convert all the xml files into a json and save it for ease of use
    annotations = {os.path.splitext(os.path.basename(xml_file))[0] : xml_to_dict(xml_file) for xml_file in xml_files}

    json_path = './jsons/annotations.json'

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=4, sort_keys=True)

    logger.info(f"Saved annotations as a json at {json_path}")


    # download and save the youtube videos
    video_path = os.path.join(data_path, 'videos')

    make_dir(video_path)

    age_restricted = [download_youtube_video(annotations[name]['url'], filename=name, path=video_path) for name in annotations]
    age_restricted = list(filter(None, age_restricted))

    # separately save the list of age restricted videos
    with open('./jsons/age_restricted_videos.json', 'w', encoding='utf-8') as f:
        json.dump(age_restricted, f, ensure_ascii=False, indent=4, sort_keys=True)