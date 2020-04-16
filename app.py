from flask import Flask,render_template,request
import Caption_Generator

app=Flask(__name__)

@app.route('/')
def caption():
	return render_template("index.html")

@app.route('/',methods=["POST"])
def get_image():

	if request.method=="POST":
		
		f=request.files['photo']
		file_path="static/"+f.filename
		f.save(file_path)

		caption=Caption_Generator.predict_captions(file_path)

		pred={"image":file_path,"generated_caption":caption}

	return render_template("index.html",predictions=pred)

if __name__=='__main__':
	app.run(debug=True)