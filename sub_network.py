import torch
import torch.nn as nn

class enc_conv_im64(nn.Module):
    def __init__(self, z_dim, y_dim, imCh=1, ngf=32, h_dim=256, useBias=True):
        super(enc_conv_im64, self).__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.imCh = imCh
        self.ngf = ngf

        self.conv1 =nn.Conv2d(imCh, ngf, (4, 4), stride=2, padding=1, bias=useBias)
        self.conv2 = nn.Conv2d(ngf, ngf, (4, 4), stride=2, padding=1, bias=useBias)
        self.conv3 = nn.Conv2d(ngf, (ngf*2), (4, 4), stride=2, padding=1, bias=useBias)
        self.conv4 = nn.Conv2d((ngf*2), (ngf*2), (4, 4), stride=2, padding=1, bias=useBias)

        self.fc1 = nn.Linear((ngf*2) * 4 * 4, h_dim)
        self.fc2_z = nn.Linear(h_dim, z_dim)
        self.fc2_y = nn.Linear(h_dim, y_dim)

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x.view(x.size(0), self.imCh, 64, 64)
        h = self.act( self.conv1(h) )
        h = self.act( self.conv2(h) )
        h = self.act( self.conv3(h) )
        h = self.act( self.conv4(h) )
        h = h.view(x.size(0), (self.ngf*2) * 4 * 4)
        h = self.act( self.fc1(h) )

        h_z = self.fc2_z(h)
        z = h_z.view(x.size(0), self.z_dim)
        y = self.log_softmax(self.fc2_y(h))

        return z, y


class dec_conv_im64(nn.Module):
    def __init__(self, z_dim, y_dim, imCh=1, ngf=32, h_dim=256, useBias=True):
        super(dec_conv_im64, self).__init__()
        self.ngf = ngf

        self.fc1 = nn.Linear(z_dim+y_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, (ngf*2) * 4 * 4)

        self.conv0 = nn.ConvTranspose2d((ngf * 2), (ngf * 2), (4, 4), stride=2, padding=1, bias=useBias)
        self.conv1 = nn.ConvTranspose2d((ngf*2), ngf, (4, 4), stride=2, padding=1, bias=useBias)
        self.conv2 = nn.ConvTranspose2d(ngf, ngf, (4, 4), stride=2, padding=1, bias=useBias)
        self.conv3 = nn.ConvTranspose2d(ngf, imCh, (4, 4), stride=2, padding=1, bias=useBias)

        self.act = nn.ReLU(inplace=True)

    def forward(self, z, y):

        h_z = z.view(z.size(0), -1)
        h_y = y.view(y.size(0), -1)
        h = torch.cat((h_z, h_y), 1)

        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        h = h.view(z.size(0), self.ngf*2, 4, 4)

        h = self.act( self.conv0(h) )
        h = self.act( self.conv1(h) )
        h = self.act( self.conv2(h) )
        h = self.conv3(h)

        return h



class enc_conv_im32(nn.Module):
    def __init__(self, z_dim, y_dim, imCh=1, ngf=32, h_dim=256, useBias=True):
        super(enc_conv_im32, self).__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.imCh = imCh
        self.ngf = ngf

        self.conv1 =nn.Conv2d(imCh, ngf, (4, 4), stride=2, padding=1, bias=useBias)
        self.conv2 = nn.Conv2d(ngf, ngf, (4, 4), stride=2, padding=1, bias=useBias)
        self.conv3 = nn.Conv2d(ngf, (ngf*2), (4, 4), stride=2, padding=1, bias=useBias)

        self.fc1 = nn.Linear((ngf*2) * 4 * 4, h_dim)
        self.fc2_z = nn.Linear(h_dim, z_dim)
        self.fc2_y = nn.Linear(h_dim, y_dim)

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x.view(x.size(0), self.imCh, 32, 32)
        h = self.act( self.conv1(h) )
        h = self.act( self.conv2(h) )
        h = self.act( self.conv3(h) )
        h = h.view(x.size(0), (self.ngf*2) * 4 * 4)
        h = self.act( self.fc1(h) )

        h_z = self.fc2_z(h)
        z = h_z.view(x.size(0), self.z_dim)
        y = self.log_softmax(self.fc2_y(h))

        return z, y


class dec_conv_im32(nn.Module):
    def __init__(self, z_dim, y_dim, imCh=1, ngf=32, h_dim=256, useBias=True):
        super(dec_conv_im32, self).__init__()
        self.ngf = ngf

        self.fc1 = nn.Linear(z_dim+y_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, (ngf*2) * 4 * 4)

        self.conv1 = nn.ConvTranspose2d((ngf*2), ngf, (4, 4), stride=2, padding=1, bias=useBias)
        self.conv2 = nn.ConvTranspose2d(ngf, ngf, (4, 4), stride=2, padding=1, bias=useBias)
        self.conv3 = nn.ConvTranspose2d(ngf, imCh, (4, 4), stride=2, padding=1, bias=useBias)

        self.act = nn.ReLU(inplace=True)

    def forward(self, z, y):

        h_z = z.view(z.size(0), -1)
        h_y = y.view(y.size(0), -1)
        h = torch.cat((h_z, h_y), 1)

        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        h = h.view(z.size(0), self.ngf*2, 4, 4)

        h = self.act( self.conv1(h) )
        h = self.act( self.conv2(h) )
        h = self.conv3(h)

        return h