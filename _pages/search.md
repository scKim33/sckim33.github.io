---
layout: page
title: "Search"
permalink: /search/
sitemap: false
sidebar:
    nav: "sidebar-category"
---

<div id="search-container">
  <input type="text" id="search-input" placeholder="Type to search...">
  <ul id="results-container"></ul>
</div>

<script src="{{ '/assets/js/simple-jekyll-search.min.js' | relative_url }}"></script>
<script>
  SimpleJekyllSearch({
    searchInput: document.getElementById('search-input'),
    resultsContainer: document.getElementById('results-container'),
    json: '{{ '/search.json' | relative_url }}',
    searchResultTemplate: '<li><a href="{url}">{title}</a></li>',
    noResultsText: 'No results found'
  })
</script>
