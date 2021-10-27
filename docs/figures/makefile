all: pdf clean

pdf:
	pdflatex *.tex

static/elo.png:
	pdflatex elo.tex
	convert -density 300 elo.pdf -quality 90 static/elo.png
	convert static/elo.png -channel RGB -negate static/elo_dark.png
	
clean:
	- rm -f Rplots.pdf
	- rm -f *.log
	- rm -f *.soc
	- rm -f *.toc
	- rm -f *.aux
	- rm -f *.out
	- rm -f main.idx
	- rm -f *.bbl
	- rm -f *.bbg
	- rm -f *.dvi
	- rm -f *.blg
	- rm -f *.lof
	- rm -f *.nav
	- rm -f *.snm
	- rm -f *~
