var express = require('express');
var router = express.Router();
var inputController = require('../app/controllers/inputController');

router.route('/curves/:id')
  .get(inputController.getCurves);

router.route('/delete/:id')
  .delete(inputController.delete);

router.route('/random')
  .post(inputController.createRandom);

router.route('/:id')
  .get(inputController.findOne);



module.exports = router;
