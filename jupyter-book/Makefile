all: html

html:
	jupyter-book build .
	rm -rf _build/html/slides _build/html/images && cp -r ../images _build/html

clean:
	# keep jupyter-cache cache folder
	jupyter-book clean .

full-clean:
	rm -rf _build
