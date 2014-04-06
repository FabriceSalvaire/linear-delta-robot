####################################################################################################
#
# LaTeX Makefile
#
####################################################################################################

# JOB_NAME = 

density = 200 # dpi

MASTER = figures
TEXFILES =  $(wildcard *.tex)
figures_pdf =  $(wildcard images/*.pdf)
figures_png =  $(patsubst %.pdf, %.png, ${figures_pdf})

# VPATH = parts:packages

####################################################################################################
#
# LaTeX options
#
####################################################################################################

#INTERACTION=
#INTERACTION=-interaction=batchmode
INTERACTION=-interaction=nonstopmode

LATEX = pdflatex ${INTERACTION} -shell-escape

####################################################################################################
#
# Rules
#
####################################################################################################

all: pdf ${figures_png}

# force
f:
	touch ${MASTER}.tex
	$(MAKE)	all

pdf: ${MASTER}.pdf

${MASTER}.pdf: ${TEXFILES}
	${LATEX} ${MASTER}

%.png : %.pdf
	convert -density ${density} -transparent white $< $@

fast: ${TEXFILES}
	${LATEX} ${MASTER}

clean:
	-rm -f *.aux *.auxlock *.log images/*.log images/*.dpth \
	images/*.pdf images/*.png \
	${MASTER}.pdf

####################################################################################################

.PHONY: f all clean link

####################################################################################################
#
# End
#
####################################################################################################
