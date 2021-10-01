var express     = require('express'),
    path        = require('path'),
    favicon     = require('serve-favicon'),
    logger      = require('morgan'),
    cookieParser = require('cookie-parser'),
    bodyParser  = require('body-parser'),
    mongoose    = require('mongoose'),
    db          = require('./app/models/db'),
    cors        = require('cors');

// ROUTE REGISTERING
var index = require('./routes/index');
var input = require('./routes/input');
var curve = require('./routes/curve');
var output = require('./routes/output');

// APP INIT
var app = express();

app.set('view engine', 'ejs');

app.use(cors());
app.use(logger('dev'));
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: false }));
app.use(cookieParser());


// ROUTES USED
app.use('/', index);
app.use('/input', input);
app.use('/curve', curve);
app.use('/output', output);

// CATCH ERROR 
app.use(function(req, res, next) {
  var err = new Error('Not Found');
  err.status = 404;
  next(err);
});

// ERROR HANDLER
app.use(function(err, req, res, next) {
  res.locals.message = err.message;
  res.locals.error = req.app.get('env') === 'development' ? err : {};

  // RENDER ERROR PAGE
  res.status(err.status || 500);
  res.render('error');
});

module.exports = app;
