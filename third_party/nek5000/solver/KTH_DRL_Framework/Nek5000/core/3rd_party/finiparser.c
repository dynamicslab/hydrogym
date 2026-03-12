#include <ctype.h>
#include "iniparser.h"

#ifndef FNAME_H
#define FNAME_H

/*
   FORTRAN naming convention
     default      cpgs_setup, etc.
     -DUPCASE     CPGS_SETUP, etc.
     -DUNDERSCORE cpgs_setup_, etc.
*/

#ifdef UPCASE
#  define FORTRAN_NAME(low,up) up
#else
#ifdef UNDERSCORE
#  define FORTRAN_NAME(low,up) low##_
#else
#  define FORTRAN_NAME(low,up) low
#endif
#endif

#endif

#define finiparser_dump           FORTRAN_NAME(finiparser_dump,  FINIPARSER_DUMP)
#define finiparser_getPair        FORTRAN_NAME(finiparser_getpair,  FINIPARSER_GETPAIR)
#define finiparser_load           FORTRAN_NAME(finiparser_load,  FINIPARSER_LOAD)
#define finiparser_free           FORTRAN_NAME(finiparser_free,  FINIPARSER_FREE)
#define finiparser_getString      FORTRAN_NAME(finiparser_getstring,  FINIPARSER_GETSTRING)
#define finiparser_getDictEntries FORTRAN_NAME(finiparser_getdictentries,  FINIPARSER_GETDICTENTRIES)
#define finiparser_getBool        FORTRAN_NAME(finiparser_getbool,  FINIPARSER_GETBOOL)
#define finiparser_find           FORTRAN_NAME(finiparser_find,  FINIPARSER_FIND)
#define finiparser_getDbl         FORTRAN_NAME(finiparser_getdbl,  FINIPARSER_GETDBL)
#define finiparser_getToken       FORTRAN_NAME(finiparser_gettoken,  FINIPARSER_GETTOKEN)
#define finiparser_findTokens     FORTRAN_NAME(finiparser_findtokens,  FINIPARSER_FINDTOKENS)

#define ntokenmax  100 

static dictionary *dic=NULL;
static char *token[ntokenmax];


char *addchar0(char *str,int str_len)
{
    int i, real_len;
    char *newstr;

    /* strip trailing blanks in datarep */
    if (str <= (char *) 0) {
        return NULL;
    }
    for (i=str_len-1; i>=0; i--) if (str[i] != ' ') break;
    if (i < 0) {
        return NULL;
    }
    real_len = i + 1;

    newstr = (char *) malloc((real_len+1)*sizeof(char));
    strncpy(newstr, str, real_len);
    newstr[real_len] = '\0';
    return newstr;
}

void finiparser_dump()
{
    if(dic != NULL) iniparser_dump(dic,stdout);
    return;
}

void finiparser_find(int* out,char *key,int* ifnd,int key_len)
{
    int tmp;
    *ifnd = 0;
    char *key_;

    key_ = addchar0(key,key_len);
    tmp = iniparser_find_entry(dic,key_);
    if (tmp == 1) {
       *out = tmp;
       *ifnd = 1;
    }
    free(key_);
    return;
}

void finiparser_getDictEntries(int *n)
{
    *n = dic->n;
    return;
}


void finiparser_getPair(char *key,char *val,int *id,int *ifnd,int key_len, int val_len)
{
    *ifnd = 0;
    int i;
    int real_key_len = 0;
    int real_val_len = 0;

    if (*id > dic->n) return;
    for (i=0; i<key_len; i++) key[i] = ' ';
    for (i=0; i<val_len; i++) val[i] = ' ';

    real_key_len = strlen(dic->key[*id-1]);
    if(dic->val[*id-1] != NULL) real_val_len = strlen(dic->val[*id-1]);

    if(real_key_len > key_len) return;
    if(real_val_len > val_len) return;

    strncpy(key,dic->key[*id-1],real_key_len);
    strncpy(val,dic->val[*id-1],real_val_len);

    *ifnd = 1;
    return;
}

void finiparser_load(char * fname,int* ierr,int fname_len)
{
    *ierr = 0;
    char *fname_;

    fname_ = addchar0(fname,fname_len); 
    dic = iniparser_load(fname_); 
    if (dic == NULL) *ierr = 1;
    free(fname_);
    return;
}

void finiparser_free()
{
    dictionary_del(dic);
    return;
}

void finiparser_getString(char *out,char *key,int *ifnd,int out_len,int key_len)
{
    int i;
    const char* str;
    int real_out_len;
    char *key_;

    *ifnd = 0;
    for (i=0; i<out_len; i++) out[i] = ' ';

    key_ = addchar0(key,key_len);
    str = iniparser_getstring(dic,key_,NULL);
    if (str != NULL) {
       real_out_len = strlen(str);
       if(real_out_len <= out_len) {
         strncpy(out,str,real_out_len);
         *ifnd = 1;
       } 
    }
    free(key_);
    return;
}

void finiparser_getBool(int* out,char *key,int* ifnd,int key_len)
{
    int tmp;
    char *key_;

    *ifnd = 0;
    key_ = addchar0(key,key_len);
    tmp = iniparser_getboolean(dic,key_,-1);
    if (tmp != -1) { 
       *out = tmp;
       *ifnd = 1;
    }
    free(key_);
    return;
}

void finiparser_getDbl(double* out,char *key,int *ifnd,int key_len)
{
    const char* str;
    char *key_;

    *ifnd = 0;
    key_ = addchar0(key,key_len);
    str = iniparser_getstring(dic,key_,NULL);
    if (str != NULL) {
       *out = atof(str);
       *ifnd = 1;
    }
    free(key_);
    return;
}

void finiparser_getToken(char *out,int *id,int out_len)
{
    int real_out_len, i;

    for (i=0; i<out_len; i++) out[i] = ' ';
    if(*id > ntokenmax) return;
    real_out_len = strlen(token[*id-1]);
    if(real_out_len <= out_len) strncpy(out,token[*id-1],real_out_len);
    return;
}

void finiparser_findTokens(char *key, char *delim, int *icounter,int key_len,int delim_len)
{
    const char *str;
    char *newstr;
    char *d, *key_;
    int i;

    *icounter = 0;

    d = addchar0(delim,delim_len);
    key_ = addchar0(key,key_len); 
    str = iniparser_getstring(dic,key_,NULL);
    free(key_);
    if (str == NULL) return;

    newstr = (char *) malloc((strlen(str)+1)*sizeof(char));
    strncpy(newstr,str,strlen(str)+1);

    i = 0;    
    token[i] = strtok(newstr,d);
    while (token[i] != NULL && i <= ntokenmax-1) {
           strstrip(token[i]);
           token[++i] = strtok(NULL,d);
    }
    *icounter = i;
    free(d);  

    return;
}
