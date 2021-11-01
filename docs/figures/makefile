all: pdf clean

pdf:
	pdflatex *.tex

../src/assets/elo.png:
	pdflatex elo.tex
	convert -density 300 elo.pdf -quality 90 ../src/assets/elo.png
	convert ../src/assets/elo.png -channel RGB -negate ../src/assets/elo_dark.png
	
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
