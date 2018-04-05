import logging

class Logger(object):
	"""Logging information"""
	levels = {"debug":10,"info":20,"warning":30,"error":40,"critical":50}
	def __init__(self, name, filename, level):
		super(Logger, self).__init__()
		self.name = name
		self.filename = filename
		self.level = self.levels[level]

	def log(self):
		"""
		Return a logger object with name, level and filename attributes of the object.
		"""
		logger_object = logging.getLogger(self.name)
		logger_object.setLevel(self.level)
		formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
		file = logging.FileHandler(self.filename)
		file.setFormatter(formatter)
		logger_object.addHandler(file)
		return logger_object