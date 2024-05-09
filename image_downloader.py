# from simple_image_download import simple_image_download as simp 

# response = simp.simple_image_download

# response().download(keywords="cars with license plates, vehicle with license plate", limit=50)





query_string = "Different countries people's house room interiors (including luxurious, middle class and poor)"


from bing_image_downloader import downloader

downloader.download(query_string, limit = 200,  output_dir = 'dataset', 
adult_filter_off = True, force_replace = False, timeout = 60, verbose = True)
