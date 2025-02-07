---
title: "Books & Lectures"
layout: archive
permalink: /categories/books_lectures/
author_profile: true
---

{% assign posts = site.categories['books_lectures']%}
{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}