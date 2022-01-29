'use strict';

const pug = require('pug');

function pugCompile(data) {
  return pug.compile(data.text, {
    filename: data.path
  });
}

function pugRenderer(data, locals) {
  return pugCompile(data)(locals);
}

pugRenderer.compile = pugCompile;

module.exports = pugRenderer;
