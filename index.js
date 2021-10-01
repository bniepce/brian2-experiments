var express = require('express');
var router = express.Router();
var inputController = require('../app/controllers/inputController');

router.route('/')
  .get(inputController.index)
  .post(inputController.create);

module.exports = router;
