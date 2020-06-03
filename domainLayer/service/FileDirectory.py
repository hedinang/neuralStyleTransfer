import os


class FileDirectory:
    def __init__(self, data):
        self.fileList = []
        for key in data.keys():
            if key == "type":
                self.type = data["type"]
                if self.type == "input":
                    self.directory = "/home/dung/Project/Django/Django/resource/input/contents"
                else:
                    self.directory = "/home/dung/Project/Django/Django/resource/input/styles"
            if key == "directory":
                self.directory = data["directory"]

    def findFileDirectory(self):
        for r, d, f in os.walk(self.directory):
            self.fileList.append(f)
        response = {
            "directories": self.fileList
        }
        return response
