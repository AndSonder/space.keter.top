#!/bin/bash
hexo clean
hexo g
export HEXO_ALGOLIA_INDEXING_KEY=e583ae17b10344887930c19a6ec47e2d
hexo algolia
hexo d

