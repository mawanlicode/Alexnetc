# standard makefile for simple app

PROJ=mnist

SYSLIB=-lm
MYLIB=
CFLAGS=
CC=gcc

OBJDIR=./obj
SRCS=$(wildcard *.c)
OBJS=$(patsubst %.c, $(OBJDIR)/%.o, $(SRCS))

$(PROJ): $(OBJS)
	$(CC) -o $@ $^ $(CFLAGS) $(SYSLIB) $(MYLIB)

$(OBJDIR)/%.o: %.c
	mkdir -p $(OBJDIR)
	$(CC) -c $< -o $@

clean:
	rm -f $(PROJ) $(OBJDIR)/*.o
