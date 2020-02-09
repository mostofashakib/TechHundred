var express        = require("express"),
    app            = express(),
    bodyParser     = require("body-parser"),
    cookieParser   = require("cookie-parser"),
    methodOverride = require("method-override");
    
app.use(bodyParser.urlencoded({extended: true}));
app.set("view engine", "ejs");
app.use(express.static(__dirname + "/public"));
app.use(methodOverride('_method'));
app.use(cookieParser('secret'));

// Restful routing

app.get("/", function(req, res){
    res.render("index");
});

app.get("/text", callName);

function callName(req, res) { 
   var spawn = require("child_process").spawn; 
      
   var process = spawn('python',["C:/Users/19514/Desktop/hacklahoma/capture.py", req.query.message] ); 
  
   process.stdout.on('data', function(data) { 
       res.send(data.toString()); 
    } ) 
};

app.get("/vision", callName);

function callName(req, res) { 
   var spawn = require("child_process").spawn; 
      
   var process = spawn('python',["C:/Users/19514/Desktop/hacklahoma/vision.py", req.query.message] ); 
  
   process.stdout.on('data', function(data) { 
       res.send(data.toString()); 
    } ) 
};


// server Listen 
app.listen(3000, function(){
   console.log("The server has started!");
});
