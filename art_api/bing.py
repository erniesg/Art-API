from bing_image_downloader import downloader
from art_api import config



def main():
    dir = config.PATH_BING
    downloader.download('aeroplane painting', limit = 3500, output_dir = dir, adult_filter_off = False)
    downloader.download('bird painting', limit = 3500, output_dir = dir, adult_filter_off = False)
    downloader.download('boat painting', limit = 3500, output_dir = dir, adult_filter_off = False)
    downloader.download('chair painting', limit = 3500, output_dir = dir, adult_filter_off = False)
    downloader.download('cow painting', limit = 3500, output_dir = dir, adult_filter_off = False)
    downloader.download('diningtable painting', limit = 3500, output_dir = dir, adult_filter_off = False)
    downloader.download('dog painting', limit = 3500, output_dir = dir, adult_filter_off = False)
    downloader.download('horse painting', limit = 3500, output_dir = dir, adult_filter_off = False)
    downloader.download('sheep painting', limit = 3500, output_dir = dir, adult_filter_off = False)
    downloader.download('train painting', limit = 3500, output_dir = dir, adult_filter_off = False)

if __name__ == '__main__':
    try:
        while True:
            main()
    except KeyboardInterrupt:
        print('\nGoodbye!')
        sys.exit(0)