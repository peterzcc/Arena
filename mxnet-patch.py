import sys
import os
import shutil
import argparse
import glob

_dirname = os.path.dirname(os.path.realpath(__file__))


def update_extension_header(mshadow_path, new_headers):
    extension_path= os.path.join(mshadow_path, "extension.h")
    backup_path = os.path.join(mshadow_path, "extension.h.arena.bak")
    if os.path.exists(backup_path):
        shutil.move(backup_path, extension_path)
        print "[mshadow-ext] Find previous backup file, moving %s to %s" \
              % (backup_path, extension_path)
    shutil.move(extension_path, backup_path)
    print "[mshadow-ext] moving %s to %s" %(extension_path, backup_path)
    extf = open(extension_path, 'w')
    backupf = open(backup_path, 'r')
    for line in backupf:
        if "#endif" not in line:
            extf.write(line)
        else:
            for path in new_headers:
                path = os.path.basename(path)
                extf.write('#include "./extension/%s"\n' %path)
            extf.write(line)

def recover_extension_header(mshadow_path, new_headers):
    extension_path = os.path.join(mshadow_path, "extension.h")
    backup_path = os.path.join(mshadow_path, "extension.h.arena.bak")
    shutil.move(backup_path, extension_path)
    print "[mshadow-ext] moving %s to %s" % (backup_path, extension_path)

parser = argparse.ArgumentParser(description='Script to install the patch for MXNet.')
parser.add_argument('-p', '--path', required=True, type=str, help='Path of MXNet.')
parser.add_argument('-t', '--type', required=True, type=str,
                    choices=['install', 'uninstall'], help='Install or uninstall the extension')
args, unknown = parser.parse_known_args()
mxnet_path = os.path.realpath(args.path)
mxnet_operator_path = os.path.join(mxnet_path, "src", "operator")
mshadow_path = os.path.join(mxnet_path, "mshadow", "mshadow")
mshadow_extention_path = os.path.join(mshadow_path, "extension")

if 'install' == args.type:
    print 'Installing MXNet Extensions'

    for ele in sorted(glob.glob("mxnet-extension/*.h") + glob.glob("mxnet-extension/*.cu") + \
            glob.glob("mxnet-extension/*.cc")):
        shutil.copy(ele, mxnet_operator_path)
        print '[mxnet-ext]: copying %s to %s' % (ele, mxnet_operator_path)
    for ele in sorted(glob.glob("mxnet-extension/mshadow-extension/*.h")):
        shutil.copy(ele, mshadow_extention_path)
        print '[mshadow-ext]: copying %s to %s' % (ele, mshadow_extention_path)
    update_extension_header(mshadow_path, glob.glob("mxnet-extension/mshadow-extension/*.h"))
elif 'uninstall'  == args.type:
    print 'Removing MXNet Extensions'
    for ele in sorted(glob.glob("mxnet-extension/*.h") + glob.glob("mxnet-extension/*.cu") + \
        glob.glob("mxnet-extension/*.cc")):
        target_path = os.path.join(mxnet_operator_path, os.path.basename(ele))
        os.remove(target_path)
        print '[mxnet-ext]: removing %s' % target_path
    for ele in sorted(glob.glob("mxnet-extension/mshadow-extension/*.h")):
        target_path = os.path.join(mshadow_extention_path, os.path.basename(ele))
        os.remove(target_path)
        print '[mshadow-ext]: removing %s' % target_path
    recover_extension_header(mshadow_path, glob.glob("mxnet-extension/mshadow-extension/*.h"))
else:
    raise NotImplementedError