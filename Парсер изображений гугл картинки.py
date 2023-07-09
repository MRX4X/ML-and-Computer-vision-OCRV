from icrawler.builtin import GoogleImageCrawler

google_crawler = GoogleImageCrawler(storage={'root_dir': '/home/user/Изображения/Хопперы'})

print('Сколько нужно спарсить фото?')
count_foto = int(input())

print('По какому запросу нужно парсить?')
requests_foto = str(input())

google_crawler.crawl(keyword=requests_foto, max_num=count_foto)