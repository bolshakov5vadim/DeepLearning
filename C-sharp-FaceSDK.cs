using System;
using System.IO;
using System.Threading;
using System.Collections.Generic;

using CommandLine;

using VDT.FaceRecognition.SDK;//библиотеки подключили



class Options//этот класс содержит конфиг главного сервиса библиотеки. А также базовые get/set
{
	[Option("config_dir", Default = "../../../conf/facerec", HelpText = "Path to config directory.")]
	public string config_dir { get; set; }

	[Option("license_dir", Default = null, HelpText = "Path to license directory [optional].")]
	public string license_dir { get; set; }

	[Option("database_dir", Default = "../../base", HelpText = "Path to database directory.")]
	public string database_dir { get; set; }

	[Option("method_config", Default = "recognizer_latest_v100.xml", HelpText = "Recognizer config file.")]
	public string method_config { get; set; }

	[Option("recognition_far_threshold", Default = 1e-6f, HelpText = "Recognition FAR threshold.")]
	public float recognition_far_threshold { get; set; }

	[Option("frame_fps_limit", Default = 25f, HelpText = "Frame fps limit.")]
	public float frame_fps_limit { get; set; }

	[Value(0, MetaName = "video_sources", HelpText = "List of video sources (id of web-camera, url of rtsp stream or path to video file)")]
	public IEnumerable<string> video_sources { get; set; }
};

class VideoRecognitionDemo
{

	static int Main(string[] args)
	{

			Console.WriteLine("");

			Options options = new Options();//ОБЪЕКТ ConfigCTX, для создания 
			CommandLine.Parser.Default.ParseArguments<Options>(args)//метод, читающий аргументы
				.WithParsed<Options>(opts => options = opts)
				.WithNotParsed<Options>(errs => error = true);

			string config_dir = options.config_dir;
			string license_dir = options.license_dir;
			string database_dir = options.database_dir;
			string method_config = options.method_config;//КОНФИГ VideoWorker
			float recognition_far_threshold = options.recognition_far_threshold;
			float frame_fps_limit = options.frame_fps_limit;
			List<string> video_sources = new List<string>(options.video_sources);

			// СИСТЕМА ЛОВЛИ ОШИБОК АРГУМЕНТОВ
			if(config_dir == string.Empty){throw new Exception("Error! config_dir is empty.");}
			if(database_dir == string.Empty) {throw new Exception("Error! database_dir is empty.");}
			if(method_config == string.Empty) {throw new Exception("Error! method_config is empty.");}
			if(recognition_far_threshold <= 0) {throw new Exception("Error! Failed recognition far threshold.");}

			List<ImageAndDepthSource> sources = new List<ImageAndDepthSource>();
			List<string> sources_names = new List<string>();


//#if !DEPTH_LIVENESS for all video_sources[] СДЕЛАТЬ sources_names.Add "OpenCvS source {0}", i И sources.Add(new OpencvSource(video_sources[i]));
//КАСТОМИЗАЦИЯ И ДОБАВЛЕНИЕ ТЕХНОЛОГИЙ ЕСЛИ НАДО
//for(int i = 0; i < sources_names.Count; ++i){Console.WriteLine("  {0}", sources_names[i]);}



			// СОЗДАЕМ СЕРВИС
			FacerecService service =
				FacerecService.createService(
					config_dir,
					license_dir);

			Recognizer recognizer = service.createRecognizer(method_config, true, false, false);//СОЗДАЕМ ГЛАВНЫЙ ОБЪЕКТ БИБЛИОТЕКИ 
			float recognition_distance_threshold = Convert.ToSingle(recognizer.getROCCurvePointByFAR(recognition_far_threshold).distance);

			Capturer capturer = service.createCapturer("common_capturer_blf_fda_front.xml");
			//ПОГНАЛИ ГЛАВНЫЙ ОБЪЕКТ

			//В этой библиотеке объект настраивается другим объектом-FacerecService.Config. 
      //FacerecService.Config vw_config = new FacerecService.Config("video_worker_fdatracker_blf_fda.xml");
			// ПРИМЕНЯЕМ ВСТРОЕННЫЕ МЕТОДЫ ТИПА overrideParameter
			//vw_config.overrideParameter("search_k", 10);
			//vw_config.overrideParameter("not_found_match_found_callback", 1);
			//vw_config.overrideParameter("downscale_rawsamples_to_preferred_size", 0);
			//vw_config.overrideParameter("depth_data_flag", 1);
			//vw_config.overrideParameter("good_light_dark_threshold", 1);
			//vw_config.overrideParameter("good_light_range_threshold", 1);

			// VideoWorker ИЗ БИБЛИОТЕКИ FACE SDK. ЧТЕНИЕ С КАМЕРЫ
			VideoWorker video_worker =
				service.createVideoWorker(
					new VideoWorker.Params()
						.video_worker_config(vw_config)
						.recognizer_ini_file(method_config)
						.streams_count(sources.Count)
						.age_gender_estimation_threads_count(sources.Count)
						.emotions_estimation_threads_count(sources.Count)
						.processing_threads_count(sources.Count)
						.matching_threads_count(sources.Count));
			//video_worker.setDatabase// ВОТ ТАКОЙ ЕЩЕ МЕТОД ЕСТЬ


			for(int i = 0; i < sources_names.Count; ++i)//ФУНКЦИЯ IMSHOW В ЦИКЛЕ
			{
				OpenCvSharp.Window window = new OpenCvSharp.Window(sources_names[i]);
				OpenCvSharp.Cv2.ImShow(sources_names[i], new OpenCvSharp.Mat(100, 100, OpenCvSharp.MatType.CV_8UC3, OpenCvSharp.Scalar.All(0)));
			}



			List<OpenCvSharp.Mat> outputs = new List<OpenCvSharp.Mat>(sources.Count);//ЛИСТ С КАРТИНКАМИ
			
			for(;;)// ПОГНАЛИ ГЛАВНЫЙ ЦИКЛ
			{
				{
					for(int i = 0; i < draw_images.Count; ++i)
					{
						OpenCvSharp.Mat drawed_im = workers[i]._draw_image;
						if(!drawed_im.Empty())
						{
							OpenCvSharp.Cv2.ImShow(sources_names[i], drawed_im);
							draw_images[i] = new OpenCvSharp.Mat();
						}
					}
				}

				int key = OpenCvSharp.Cv2.WaitKey(20);//ЧТЕНИЕ КЛАВИШИ КАЖДЫЕ 20 МС
				if(27 == key)// НУ И РЕАКЦИИ, СОБСТВЕННО
				{
					foreach(Worker w in workers)
					{
						w.Dispose();
					}
					break;
				}

				if(' ' == key)
				{
					Console.WriteLine("enable processing 0");
					video_worker.enableProcessingOnStream(0);
				}

				if(13 == key)
				{
					Console.WriteLine("disable processing 0");
					video_worker.disableProcessingOnStream(0);
				}


				if('r' == key)
				{
					Console.WriteLine("reset trackerOnStream");
					video_worker.resetTrackerOnStream(0);
				}


			}

			// ВЫКЛЮЧАЕМ СЕРВИС
			service.Dispose();
			video_worker.Dispose();
		}
		
		return 0;
	}
}



			//MUTEX-тип/объект, используется с объектами WORKER. Надо изучить, зачем они нужны
